"""
Flux + CLIP objective image optimisation example.

This script:
- Constructs a 2D u-space from 3 basis latents (3 seeds => u in [0,1]^2)
- Uses `latent_spaces_by_example.knothe_rosenblatt_surrogate_chart` to map u -> Flux latents
- Scores generated images with CLIP similarity to a scoring prompt
- Evaluates a 2D grid and saves a contour plot
- Runs SciPy DIRECT to search for the best u (budgeted evaluations)

Install optional deps:
  uv sync --group examples

Dry run (no model download; shape plumbing only):
  uv run python examples/image_optimisation.py --dry-run
"""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from latent_spaces_by_example import knothe_rosenblatt_surrogate_chart

try:
    import click
    import matplotlib.pyplot as plt
    import torch
    from diffusers import FluxPipeline
    from PIL import Image, ImageDraw, ImageFont
    from scipy.optimize import Bounds, direct
    from tqdm.auto import tqdm
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    print(
        "Optional dependencies missing.\n\n"
        "Install (example):\n"
        "  uv sync --group examples\n\n"
        "Then try:\n"
        "  uv run python examples/image_optimisation.py --dry-run\n\n"
        "Notes:\n"
        "  - For CUDA, install a CUDA-enabled torch build per PyTorch instructions.\n"
        "  - You may also need to `huggingface-cli login` for gated models.\n"
    )
    raise SystemExit(2)

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

GRID_TILE_PX = 96


def _get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_clip_scorer(prompt: str, device: str) -> Callable[[Any], float]:
    """
    Makes a CLIP scorer that scores images with the specified prompt.

    Defines the objective we are searching over.
    """
    model_id = "openai/clip-vit-base-patch32"
    model = (
        CLIPModel.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
        )
        .to(device)
        .eval()
    )
    processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def score(image: Any) -> float:
        inputs = processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sim = (image_features * text_features).sum(dim=-1).item()
        return float(sim)

    return score


def _make_flux_pipeline(*, device: str) -> FluxPipeline:
    """
    Makes a Flux pipeline and offloads it to the specified device.

    IMPORTANT: Flux is large. Doing `pipe.to("cuda")` eagerly moves all weights onto GPU and
    will often OOM. Prefer CPU offload on CUDA.

    Defines the generative model we are searching over.
    """
    if device == "cuda":
        torch_dtype = torch.float16
        pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        else:
            # Fallback: may OOM, but keeps behavior for older diffusers versions.
            pipe = pipe.to(device)
    else:
        torch_dtype = torch.float32
        pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)
        pipe = pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe


def _latent_shape_for_flux(
    pipe: FluxPipeline, width: int, height: int
) -> tuple[int, int, int]:
    """
    Returns the latent shape of the latent space of the Flux pipeline.

    Useful if you want to generate latents without inverting them from images.
    """
    scale = int(getattr(pipe, "vae_scale_factor", 8))
    h = 2 * (height // (scale * 2))
    w = 2 * (width // (scale * 2))
    c = int(getattr(pipe.transformer.config, "in_channels", 64)) // 4
    return (c, h, w)


def _save_u_grid_image_montage(
    *,
    image_paths: list[list[str]],
    scores: list[list[float]],
    out_path: Path,
    tile_px: int = GRID_TILE_PX,
    border_colors: list[list[tuple[int, int, int] | None]] | None = None,
) -> None:
    """
    Save a tiled montage of grid images (u2 rows, u1 columns), with score overlay.
    """
    grid_size = len(image_paths)
    if grid_size == 0 or any(len(row) != grid_size for row in image_paths):
        raise ValueError("image_paths must be a non-empty square list[list[str]]")
    if len(scores) != grid_size or any(len(row) != grid_size for row in scores):
        raise ValueError(
            "scores must be a square list[list[float]] matching image_paths"
        )
    if border_colors is not None:
        if len(border_colors) != grid_size or any(
            len(row) != grid_size for row in border_colors
        ):
            raise ValueError(
                "border_colors must be a square list[list[tuple|None]] matching image_paths"
            )

    font = ImageFont.load_default()
    pad = 2
    w_canvas = grid_size * tile_px + (grid_size + 1) * pad
    h_canvas = grid_size * tile_px + (grid_size + 1) * pad
    canvas = Image.new("RGB", (w_canvas, h_canvas), color=(16, 16, 16))
    draw = ImageDraw.Draw(canvas)

    scores_arr = np.asarray(scores, dtype=np.float64)
    best_r, best_c = np.unravel_index(int(np.argmax(scores_arr)), scores_arr.shape)

    for r in range(grid_size):
        for c in range(grid_size):
            p = image_paths[r][c]
            s = float(scores[r][c])
            x0 = pad + c * (tile_px + pad)
            # Align montage with matplotlib axes conventions:
            # - In plots, u2 increases upwards.
            # - In images, y increases downwards.
            # So we draw higher-u2 rows nearer the top by flipping r for placement.
            y0 = pad + (grid_size - 1 - r) * (tile_px + pad)

            img = Image.open(p).convert("RGB").resize((tile_px, tile_px))
            canvas.paste(img, (x0, y0))

            # Score overlay with a small dark box for legibility
            label = f"{float(s):.3f}"
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            box = (x0, y0, x0 + tw + 6, y0 + th + 4)
            draw.rectangle(box, fill=(0, 0, 0))
            draw.text((x0 + 3, y0 + 2), label, fill=(255, 255, 255), font=font)

            outline = None
            if border_colors is not None:
                outline = border_colors[r][c]
            elif r == best_r and c == best_c:
                # Default: highlight best tile.
                outline = (0, 255, 0)
            if outline is not None:
                draw.rectangle(
                    (x0, y0, x0 + tile_px - 1, y0 + tile_px - 1),
                    outline=outline,
                    width=3,
                )

    canvas.save(out_path)


def _direct_search(
    *,
    objective: Callable[[np.ndarray], float],
    max_evals: int,
    u_dims: int,
) -> tuple[np.ndarray, float, list[list[float]], list[float]]:
    """
    Runs DIRECT optimisation to find the best u in the u-space.

    Returns the best u, the best score, the observed u values, and the observed scores.
    """
    observed_u: list[list[float]] = []
    observed_scores: list[float] = []

    pbar = tqdm(total=int(max_evals), desc="DIRECT", unit="eval")

    def loss_fn(u: np.ndarray) -> float:
        u = np.asarray(u, dtype=np.float64)
        s = float(objective(u))
        observed_u.append([float(x) for x in u.tolist()])
        observed_scores.append(float(s))
        if pbar.n < pbar.total:
            pbar.update(1)
        return -s

    bounds = Bounds(np.zeros(u_dims), np.ones(u_dims))
    try:
        direct(
            func=loss_fn,
            bounds=bounds,
            maxiter=int(max_evals),
            maxfun=int(max_evals),
            locally_biased=False,
        )
    finally:
        pbar.close()

    best_idx = int(np.argmax(np.asarray(observed_scores)))
    best_u = np.asarray(observed_u[best_idx], dtype=np.float64)
    best_s = float(observed_scores[best_idx])
    return best_u, best_s, observed_u, observed_scores


@dataclass(frozen=True)
class EvalResult:
    u: list[float]
    score: float
    image_path: str | None
    wall_time_s: float


class History:
    def __init__(self, outdir: Path, prefix: str):
        self.history: list[EvalResult] = []
        self.best_score = -float("inf")
        self.best_u: np.ndarray | None = None
        self.best_image_path: str | None = None
        self.outdir = outdir
        self.prefix = prefix

    def update(
        self,
        *,
        u: np.ndarray,
        score: float,
        wall_time_s: float,
        image: Image.Image | None,
        image_path: str | None,
    ) -> None:
        self.history.append(
            EvalResult(
                u=[
                    float(x)
                    for x in np.asarray(u, dtype=np.float64).reshape(-1).tolist()
                ],
                score=score,
                image_path=image_path,
                wall_time_s=wall_time_s,
            )
        )

        if image is not None and score > self.best_score:
            self.best_score = score
            self.best_u = u
            self.best_image_path = str(self.outdir / f"{self.prefix}_best_image.png")
            image.save(self.best_image_path)

        with open(
            self.outdir / f"{self.prefix}_history.json", "w", encoding="utf-8"
        ) as f:
            json.dump([r.__dict__ for r in self.history], f, indent=2)


@click.command(context_settings={"show_default": True})
@click.option("--generation-prompt", type=str, default="a high quality photo of a car")
@click.option("--score-prompt", type=str, default="a red sports car")
@click.option("--negative-prompt", type=str, default="")
@click.option("--width", type=int, default=256)
@click.option("--height", type=int, default=256)
@click.option("--guidance-scale", type=float, default=3.5)
@click.option("--num-inference-steps", type=int, default=5)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto")
@click.option("--random-seed", type=int, default=0)
@click.option("--grid-size", type=int, default=9)
@click.option("--direct-evals", type=int, default=20)
@click.option("--outdir", type=Path, default="outputs")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--skip-grid", is_flag=True, default=False)
@click.option("--skip-direct", is_flag=True, default=False)
def main(
    *,
    generation_prompt: str,
    score_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    device: str,
    random_seed: int,
    grid_size: int,
    direct_evals: int,
    outdir: Path,
    dry_run: bool,
    skip_grid: bool,
    skip_direct: bool,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Fixed to 3 seeds -> 2D u-space
    num_seeds = 3
    u_dims = 2

    if grid_size < 2:
        raise SystemExit("--grid-size must be >= 2")
    if direct_evals < 1:
        raise SystemExit("--direct-evals must be >= 1")

    _set_seed(int(random_seed))

    # Dry-run: shape plumbing only, without loading models from HF.
    if dry_run:
        latent_shape = (16, height // 8, width // 8)
        latent_dim = int(np.prod(latent_shape))
        seed_latents = np.random.standard_normal(size=(num_seeds, latent_dim)).astype(
            np.float64
        )
        chart = knothe_rosenblatt_surrogate_chart(seed_latents)
        u = np.random.uniform(size=(1, u_dims))
        z_flat = chart.from_u_to_z(u)[0]
        print("dry-run ok")
        print("u dims:", u_dims, "(3 seeds)")
        print("latent dim:", latent_dim)
        print("z_flat shape:", z_flat.shape)
        return

    device = _get_device(device)
    pipe = _make_flux_pipeline(device=device)

    latent_shape = _latent_shape_for_flux(pipe, width=width, height=height)
    c_lat, h_lat, w_lat = latent_shape
    latent_dim = int(c_lat * h_lat * w_lat)

    # Basis latents (unpacked). Flatten for surrogate chart.
    seed_latents_unpacked = torch.randn(
        (num_seeds, c_lat, h_lat, w_lat), device="cpu", dtype=torch.float32
    )
    seed_latents = (
        seed_latents_unpacked.reshape(num_seeds, -1).numpy().astype(np.float64)
    )
    chart = knothe_rosenblatt_surrogate_chart(seed_latents)
    seed_u = np.asarray(
        chart.from_z_to_u(seed_latents), dtype=np.float64
    )  # (num_seeds, 2)

    score_fn = _make_clip_scorer(score_prompt, device=device)

    grid_history = History(outdir=outdir, prefix="grid")

    def u_to_unpacked_latents(u: np.ndarray) -> torch.Tensor:
        u = np.asarray(u, dtype=np.float64)
        if u.shape == (u_dims,):
            u = u[None, :]
        if u.shape != (1, u_dims):
            raise ValueError(f"Expected u shape (2,) or (1,2); got {u.shape}")
        z_flat = chart.from_u_to_z(u)[0]
        return torch.from_numpy(z_flat.astype(np.float32)).reshape(
            1, c_lat, h_lat, w_lat
        )

    def objective(u: np.ndarray) -> tuple[Image.Image, float, float]:
        t0 = time.time()

        latents_unpacked = u_to_unpacked_latents(u).to(device)

        if not hasattr(pipe, "_pack_latents"):
            raise ValueError("Flux pipeline does not support packing latents")

        latents = pipe._pack_latents(
            latents=latents_unpacked,
            batch_size=latents_unpacked.shape[0],
            num_channels_latents=latents_unpacked.shape[1],
            height=latents_unpacked.shape[2],
            width=latents_unpacked.shape[3],
        )

        result = pipe(
            prompt=generation_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            latents=latents,
            num_images_per_prompt=1,
        )
        image = result.images[0].convert("RGB")
        score = score_fn(image)
        dt = time.time() - t0
        return image, score, dt

    # ---- 2D grid plot (u-space) ----
    u1 = None
    u2 = None
    U1 = None
    U2 = None
    scores_arr = None
    grid_image_paths = None
    grid_scores = None
    if not skip_grid:
        u1 = np.linspace(0.0, 1.0, grid_size)
        u2 = np.linspace(0.0, 1.0, grid_size)
        U1, U2 = np.meshgrid(u1, u2)
        grid_image_paths: list[list[str]] = [
            ["" for _ in range(grid_size)] for _ in range(grid_size)
        ]
        grid_scores: list[list[float]] = [
            [float("NaN") for _ in range(grid_size)] for _ in range(grid_size)
        ]

        total = grid_size * grid_size
        pbar = tqdm(total=total, desc="Grid", unit="img")
        for r in range(grid_size):  # u2 rows
            for col in range(grid_size):  # u1 cols
                u = np.array([u1[col], u2[r]], dtype=np.float64)
                img, s, dt = objective(u)
                path = str(
                    outdir / f"grid_r{r:02d}_c{col:02d}_score_{float(s):.4f}.png"
                )
                img.save(path)

                grid_image_paths[r][col] = path
                grid_scores[r][col] = s
                grid_history.update(
                    u=u, score=s, wall_time_s=dt, image=img, image_path=path
                )
                pbar.update(1)
        pbar.close()

        scores_arr = np.asarray(grid_scores, dtype=np.float64)
        with open(outdir / "grid_scores.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "grid_size": int(grid_size),
                    "u1": u1.tolist(),
                    "u2": u2.tolist(),
                    "scores": scores_arr.tolist(),
                },
                f,
                indent=2,
            )

    # ---- DIRECT optimisation ----
    direct_history = History(outdir=outdir, prefix="direct")
    if not skip_direct:

        def objective_score_only(u: np.ndarray) -> float:
            img, s, dt = objective(u)
            direct_history.update(
                u=u, score=s, wall_time_s=dt, image=img, image_path=None
            )
            return s

        _direct_search(
            objective=objective_score_only,
            max_evals=direct_evals,
            u_dims=u_dims,
        )

    # ---- Plotting (after grid and direct) ----
    if not skip_grid:
        assert u1 is not None and u2 is not None and U1 is not None and U2 is not None
        assert (
            scores_arr is not None
            and grid_image_paths is not None
            and grid_scores is not None
        )

        # Contour with overlays: seeds + DIRECT evaluations
        plt.figure(figsize=(6, 5))
        cs = plt.contourf(U1, U2, scores_arr, levels=20)
        plt.colorbar(cs, label="CLIP score")
        plt.xlabel("u1")
        plt.ylabel("u2")
        plt.title("Flux + CLIP score over u-grid (3 seeds -> 2D)")

        # Seed locations
        plt.scatter(
            seed_u[:, 0],
            seed_u[:, 1],
            marker="*",
            s=140,
            c="white",
            edgecolors="black",
            linewidths=1.0,
            label="Seeds",
            zorder=5,
        )

        # DIRECT evaluations
        if len(direct_history.history) > 0:
            direct_u = np.asarray(
                [e.u for e in direct_history.history], dtype=np.float64
            )
            direct_scores = np.asarray(
                [e.score for e in direct_history.history], dtype=np.float64
            )
            if direct_u.ndim == 2 and direct_u.shape[1] == 2:
                plt.scatter(
                    direct_u[:, 0],
                    direct_u[:, 1],
                    c=direct_scores,
                    cmap=cs.cmap,
                    norm=cs.norm,
                    s=12,
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=0.25,
                    label="DIRECT evals",
                    zorder=7,
                )

        plt.legend(loc="best", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(outdir / "grid_scores_contour.png", dpi=200)
        plt.close()

        # Montage borders:
        # - best grid tile: green
        # - for each seed, nearest tile: red (unless it's also best)
        best_r, best_c = np.unravel_index(int(np.argmax(scores_arr)), scores_arr.shape)

        # Determine nearest cell per seed
        grid_coords = np.stack([U1, U2], axis=-1)  # (R,C,2)
        border_colors: list[list[tuple[int, int, int] | None]] = [
            [None for _ in range(grid_size)] for _ in range(grid_size)
        ]

        for k in range(seed_u.shape[0]):
            du = grid_coords - seed_u[k][None, None, :]
            dist2 = du[..., 0] ** 2 + du[..., 1] ** 2
            r0, c0 = np.unravel_index(int(np.argmin(dist2)), dist2.shape)
            border_colors[r0][c0] = (255, 0, 0)

        # Best tile overrides to green
        border_colors[int(best_r)][int(best_c)] = (0, 255, 0)

        _save_u_grid_image_montage(
            image_paths=grid_image_paths,
            scores=grid_scores,
            border_colors=border_colors,
            out_path=outdir / "grid_images_montage.png",
            tile_px=GRID_TILE_PX,
        )

    summary = {
        "u_dims": u_dims,
        "latent_shape": list(latent_shape),
        "latent_dim": latent_dim,
        "generation_prompt": generation_prompt,
        "score_prompt": score_prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "device": device,
        "grid_size": grid_size,
        "direct_evals": direct_evals,
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("done")
    print("best_score by grid search:", grid_history.best_score)
    print("best_u by grid search:", grid_history.best_u)
    if grid_history.best_image_path is not None:
        print("best_image by grid search:", grid_history.best_image_path)
    print("best_score by direct search:", direct_history.best_score)
    print("best_u by direct search:", direct_history.best_u)
    if direct_history.best_image_path is not None:
        print("best_image by direct search:", direct_history.best_image_path)


if __name__ == "__main__":
    main()

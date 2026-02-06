import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
REQUIRED_MODULES = [
    "click",
    "diffusers",
    "matplotlib",
    "PIL",
    "scipy",
    "torch",
    "tqdm",
    "transformers",
]


def _run_script(
    path: Path, args: list[str] | None = None, timeout_s: int = 60
) -> subprocess.CompletedProcess[str]:
    args = args or []
    return subprocess.run(
        [sys.executable, str(path), *args],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )


def test_basic_surrogate_chart_runs() -> None:
    proc = _run_script(EXAMPLES_DIR / "basic_surrogate_chart.py", timeout_s=30)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_image_optimisation_dry_run_runs_if_deps_installed() -> None:
    # Avoid importing the module at collection time; it has optional heavy deps.

    for mod in REQUIRED_MODULES:
        pytest.importorskip(mod)

    proc = _run_script(
        EXAMPLES_DIR / "image_optimisation.py",
        args=["--dry-run", "--device", "cpu"],
        timeout_s=60,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_grid_montage_renders_all_tiles_if_deps_installed(tmp_path: Path) -> None:
    # This is a pure unit test for montage ordering; no model download, no diffusers needed at runtime,
    # but importing the example module requires its optional deps, so we skip if missing.
    for mod in REQUIRED_MODULES:
        pytest.importorskip(mod)

    # `examples/` isn't a package, so load the module from its path.
    import importlib.util

    from PIL import Image

    module_path = EXAMPLES_DIR / "image_optimisation.py"
    spec = importlib.util.spec_from_file_location(
        "image_optimisation_example", module_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _save_u_grid_image_montage = mod._save_u_grid_image_montage

    grid_size = 2
    # Create 4 solid-color images (distinct) so missing tiles is easy to catch by dimensions only.
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
    ]

    image_paths: list[list[str]] = [
        ["" for _ in range(grid_size)] for _ in range(grid_size)
    ]
    scores: list[list[float]] = [
        [0.0 for _ in range(grid_size)] for _ in range(grid_size)
    ]
    k = 0
    for r in range(grid_size):
        for c in range(grid_size):
            img = Image.new("RGB", (64, 64), color=colors[k])
            p = tmp_path / f"img_r{r}_c{c}.png"
            img.save(p)
            image_paths[r][c] = str(p)
            scores[r][c] = float(k)
            k += 1

    out_path = tmp_path / "montage.png"
    _save_u_grid_image_montage(
        image_paths=image_paths, scores=scores, out_path=out_path, tile_px=32
    )

    out = Image.open(out_path)
    assert out.size[0] > 0 and out.size[1] > 0
    # Expected canvas: grid_size*tile_px + (grid_size+1)*pad, with pad=2 in implementation.
    expected = grid_size * 32 + (grid_size + 1) * 2
    assert out.size == (expected, expected)

def test_import() -> None:
    import latent_spaces_by_example as pkg

    assert isinstance(pkg.__version__, str)

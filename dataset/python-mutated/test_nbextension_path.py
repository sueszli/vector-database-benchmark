def test_nbextension_path():
    if False:
        i = 10
        return i + 15
    from pydeck import _jupyter_nbextension_paths
    path = _jupyter_nbextension_paths()
    assert len(path) == 1
    assert isinstance(path[0], dict)
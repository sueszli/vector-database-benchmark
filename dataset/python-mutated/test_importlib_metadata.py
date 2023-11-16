from __future__ import annotations
import pytest
pytest

def test_importlib_metadata_works() -> None:
    if False:
        print('Hello World!')
    import bokeh
    import importlib.metadata
    assert importlib.metadata is not None
from __future__ import annotations
import pytest
pytest
from pathlib import Path
from tests.support.util.api import verify_all
ALL = ('movie_path',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.movies_data', ALL))

@pytest.mark.sampledata
def test_movie_path() -> None:
    if False:
        i = 10
        return i + 15
    import bokeh.sampledata.movies_data as bsm
    assert isinstance(bsm.movie_path, Path)
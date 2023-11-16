from __future__ import annotations
import pytest
pytest
from pathlib import Path
from tests.support.util.api import verify_all
ALL = ('frontalface_default_path',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.haar_cascade', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        while True:
            i = 10
    import bokeh.sampledata.haar_cascade as bsh
    assert isinstance(bsh.frontalface_default_path, Path)
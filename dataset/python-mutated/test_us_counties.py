from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.us_counties', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        while True:
            i = 10
    import bokeh.sampledata.us_counties as bsu
    assert isinstance(bsu.data, dict)
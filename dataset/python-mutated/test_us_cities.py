from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.us_cities', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        i = 10
        return i + 15
    import bokeh.sampledata.us_cities as bsu
    assert isinstance(bsu.data, dict)
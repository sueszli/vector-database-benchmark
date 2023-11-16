from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.unemployment', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        return 10
    import bokeh.sampledata.unemployment as bsu
    assert isinstance(bsu.data, dict)
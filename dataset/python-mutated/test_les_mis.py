from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.les_mis', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        i = 10
        return i + 15
    import bokeh.sampledata.les_mis as bsl
    assert isinstance(bsl.data, dict)
    assert set(bsl.data.keys()) == {'links', 'nodes'}
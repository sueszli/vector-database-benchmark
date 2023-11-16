from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.us_states', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        print('Hello World!')
    import bokeh.sampledata.us_states as bsu
    assert isinstance(bsu.data, dict)
    assert len(bsu.data) == 51
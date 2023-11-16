from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.olympics2014', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        return 10
    import bokeh.sampledata.olympics2014 as bso
    assert isinstance(bso.data, dict)
    assert set(bso.data.keys()) == {'count', 'data', 'object'}
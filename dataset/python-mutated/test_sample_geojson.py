from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
ALL = ('geojson',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.sample_geojson', ALL))

@pytest.mark.sampledata
def test_geojson() -> None:
    if False:
        print('Hello World!')
    import bokeh.sampledata.sample_geojson as bsg
    assert isinstance(bsg.geojson, str)
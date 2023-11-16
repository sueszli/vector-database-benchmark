from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('airports', 'routes')
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.airport_routes', ALL))

@pytest.mark.sampledata
def test_airports() -> None:
    if False:
        while True:
            i = 10
    import bokeh.sampledata.airport_routes as bsa
    assert isinstance(bsa.airports, pd.DataFrame)

@pytest.mark.sampledata
def test_routes() -> None:
    if False:
        while True:
            i = 10
    import bokeh.sampledata.airport_routes as bsa
    assert isinstance(bsa.routes, pd.DataFrame)
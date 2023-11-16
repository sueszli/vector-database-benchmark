from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('sea_surface_temperature',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.sea_surface_temperature', ALL))

@pytest.mark.sampledata
def test_sea_surface_temperature() -> None:
    if False:
        return 10
    import bokeh.sampledata.sea_surface_temperature as bss
    assert isinstance(bss.sea_surface_temperature, pd.DataFrame)
    assert len(bss.sea_surface_temperature) == 19226
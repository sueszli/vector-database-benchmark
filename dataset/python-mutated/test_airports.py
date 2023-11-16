from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.airports', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        while True:
            i = 10
    import bokeh.sampledata.airports as bsa
    assert isinstance(bsa.data, pd.DataFrame)
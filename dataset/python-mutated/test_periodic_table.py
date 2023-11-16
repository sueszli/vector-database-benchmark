from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('elements',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.periodic_table', ALL))

@pytest.mark.sampledata
def test_elements() -> None:
    if False:
        while True:
            i = 10
    import bokeh.sampledata.periodic_table as bsp
    assert isinstance(bsp.elements, pd.DataFrame)
    assert len(bsp.elements) == 118
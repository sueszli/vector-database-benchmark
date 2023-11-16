from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.unemployment1948', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        for i in range(10):
            print('nop')
    import bokeh.sampledata.unemployment1948 as bsu
    assert isinstance(bsu.data, pd.DataFrame)
    assert len(bsu.data) == 69
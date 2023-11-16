from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.sample_superstore', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        print('Hello World!')
    import bokeh.sampledata.sample_superstore as bss
    assert isinstance(bss.data, pd.DataFrame)
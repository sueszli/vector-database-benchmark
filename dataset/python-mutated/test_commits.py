from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.commits', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        i = 10
        return i + 15
    import bokeh.sampledata.commits as bsc
    assert isinstance(bsc.data, pd.DataFrame)
    assert len(bsc.data) == 4916
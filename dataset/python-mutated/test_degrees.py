from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.degrees', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        print('Hello World!')
    import bokeh.sampledata.degrees as bsd
    assert isinstance(bsd.data, pd.DataFrame)
    assert len(bsd.data) == 42
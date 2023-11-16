from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('daylight_warsaw_2013',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.daylight', ALL))

@pytest.mark.sampledata
def test_daylight_warsaw_2013() -> None:
    if False:
        return 10
    import bokeh.sampledata.daylight as bsd
    assert isinstance(bsd.daylight_warsaw_2013, pd.DataFrame)
    assert len(bsd.daylight_warsaw_2013) == 365
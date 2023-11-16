from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('browsers_nov_2013', 'icons')
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.browsers', ALL))

@pytest.mark.sampledata
def test_browsers_nov_2013() -> None:
    if False:
        print('Hello World!')
    import bokeh.sampledata.browsers as bsb
    assert isinstance(bsb.browsers_nov_2013, pd.DataFrame)
    assert len(bsb.browsers_nov_2013) == 118

@pytest.mark.sampledata
def test_icons() -> None:
    if False:
        for i in range(10):
            print('nop')
    import bokeh.sampledata.browsers as bsb
    assert isinstance(bsb.icons, dict)
    assert set(bsb.icons.keys()).issubset({'Chrome', 'Firefox', 'Safari', 'Opera', 'IE'})
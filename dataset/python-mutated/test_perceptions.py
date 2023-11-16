from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('numberly', 'probly')
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.perceptions', ALL))

@pytest.mark.sampledata
def test_numberly() -> None:
    if False:
        for i in range(10):
            print('nop')
    import bokeh.sampledata.perceptions as bsp
    assert isinstance(bsp.numberly, pd.DataFrame)
    assert len(bsp.numberly) == 46

@pytest.mark.sampledata
def test_probly() -> None:
    if False:
        print('Hello World!')
    import bokeh.sampledata.perceptions as bsp
    assert isinstance(bsp.probly, pd.DataFrame)
    assert len(bsp.probly) == 46
from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('flowers',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.iris', ALL))

@pytest.mark.sampledata
def test_flowers() -> None:
    if False:
        print('Hello World!')
    import bokeh.sampledata.iris as bsi
    assert isinstance(bsi.flowers, pd.DataFrame)
    assert len(bsi.flowers) == 150
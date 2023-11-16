from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.anscombe', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        return 10
    import bokeh.sampledata.anscombe as bsa
    assert isinstance(bsa.data, pd.DataFrame)
    assert len(bsa.data) == 11
    assert list(bsa.data.columns) == ['Ix', 'Iy', 'IIx', 'IIy', 'IIIx', 'IIIy', 'IVx', 'IVy']
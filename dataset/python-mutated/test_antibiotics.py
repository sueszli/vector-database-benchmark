from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('data',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.antibiotics', ALL))

@pytest.mark.sampledata
def test_data() -> None:
    if False:
        for i in range(10):
            print('nop')
    import bokeh.sampledata.antibiotics as bsa
    assert isinstance(bsa.data, pd.DataFrame)
    assert len(bsa.data) == 16
    assert list(bsa.data.columns) == ['bacteria', 'penicillin', 'streptomycin', 'neomycin', 'gram']
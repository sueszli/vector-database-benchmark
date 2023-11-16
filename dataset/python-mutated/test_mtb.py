from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('obiszow_mtb_xcm',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.mtb', ALL))

@pytest.mark.sampledata
def test_obiszow_mtb_xcm() -> None:
    if False:
        while True:
            i = 10
    import bokeh.sampledata.mtb as bsm
    assert isinstance(bsm.obiszow_mtb_xcm, pd.DataFrame)
    assert len(bsm.obiszow_mtb_xcm) == 978
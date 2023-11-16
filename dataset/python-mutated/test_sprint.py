from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
ALL = ('sprint',)
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.sprint', ALL))

@pytest.mark.sampledata
def test_sprint() -> None:
    if False:
        i = 10
        return i + 15
    import bokeh.sampledata.sprint as bss
    assert isinstance(bss.sprint, pd.DataFrame)
    assert len(bss.sprint) == 85
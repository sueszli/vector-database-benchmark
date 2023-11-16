from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
ALL = ('fertility', 'life_expectancy', 'population', 'regions')
Test___all__ = pytest.mark.sampledata(verify_all('bokeh.sampledata.gapminder', ALL))

@pytest.mark.sampledata
@pytest.mark.parametrize('name', ['fertility', 'life_expectancy', 'population', 'regions'])
def test_data(name) -> None:
    if False:
        return 10
    import pandas as pd
    import bokeh.sampledata.gapminder as bsg
    data = getattr(bsg, name)
    assert isinstance(data, pd.DataFrame)
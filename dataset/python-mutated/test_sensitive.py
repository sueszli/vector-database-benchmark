from datetime import datetime, timedelta
import pandas as pd
import pytest
from ydata_profiling import ProfileReport

@pytest.fixture()
def df():
    if False:
        print('Hello World!')
    df = pd.DataFrame({'name': ['John Doe', 'Marco Polo', 'Louis Brandeis', 'William Douglas'], 'year': [1965, 1271, 1916, 1975], 'tf': [True, False, False, True], 'date': pd.to_datetime([datetime.now() - timedelta(days=i) for i in range(4)])})
    return df

def test_sensitive(df: pd.DataFrame):
    if False:
        i = 10
        return i + 15
    report = ProfileReport(df, sensitive=True, explorative=True)
    html = report.to_html()
    assert all((value not in html for value in df['name'].values.tolist()))
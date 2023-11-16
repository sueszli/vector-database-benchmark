import json
import pandas as pd
import pytest
from ydata_profiling import ProfileReport

@pytest.fixture
def data():
    if False:
        for i in range(10):
            print('nop')
    return pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})

def test_json(data):
    if False:
        return 10
    report = ProfileReport(data)
    report_json = report.to_json()
    data = json.loads(report_json)
    assert set(data.keys()) == {'analysis', 'time_index_analysis', 'correlations', 'duplicates', 'alerts', 'missing', 'package', 'sample', 'scatter', 'table', 'variables'}

def test_repr(data):
    if False:
        while True:
            i = 10
    report = ProfileReport(data)
    assert repr(report) == ''
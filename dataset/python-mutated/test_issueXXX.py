"""
Test for issue XXX:
https://github.com/ydataai/ydata-profiling/issues/XXX
"""
import pandas as pd
import pytest
from ydata_profiling import ProfileReport

@pytest.mark.skip()
def test_issueXXX():
    if False:
        i = 10
        return i + 15
    df = pd.read_csv('<file>')
    report = ProfileReport(df)
    _ = report.description_set
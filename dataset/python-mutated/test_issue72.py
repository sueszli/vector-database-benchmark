"""
Test for issue 72:
https://github.com/ydataai/ydata-profiling/issues/72
"""
import numpy as np
import pandas as pd
import ydata_profiling

def test_issue72_higher():
    if False:
        i = 10
        return i + 15
    df = pd.DataFrame({'A': [1, 2, 3, 3]})
    df['B'] = df['A'].apply(str)
    report = ydata_profiling.ProfileReport(df, correlations=None)
    report.config.vars.num.low_categorical_threshold = 2
    assert report.get_description().variables['A']['type'] == 'Numeric'
    assert report.get_description().variables['B']['type'] == 'Numeric'

def test_issue72_equal():
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame({'A': [1, 2, 3, 3]})
    df['B'] = df['A'].apply(str)
    report = ydata_profiling.ProfileReport(df, vars={'num': {'low_categorical_threshold': 3}}, correlations=None)
    assert report.get_description().variables['A']['type'] == 'Categorical'
    assert report.get_description().variables['B']['type'] == 'Text'

def test_issue72_lower():
    if False:
        return 10
    df = pd.DataFrame({'A': [1, 2, 3, 3, np.nan]})
    df['B'] = df['A'].apply(str)
    report = df.profile_report(correlations=None)
    report.config.vars.num.low_categorical_threshold = 10
    assert report.get_description().variables['A']['type'] == 'Categorical'
    assert report.get_description().variables['B']['type'] == 'Text'
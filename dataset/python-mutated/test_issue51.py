"""
Test for issue 51:
https://github.com/ydataai/ydata-profiling/issues/51
"""
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

def test_issue51(get_data_file):
    if False:
        while True:
            i = 10
    file_name = get_data_file('buggy1.pkl', 'https://raw.githubusercontent.com/adamrossnelson/HelloWorld/master/sparefiles/buggy1.pkl')
    df = pd.read_pickle(str(file_name))
    report = ProfileReport(df, title='Pandas Profiling Report', progress_bar=False, explorative=True)
    assert '<title>Pandas Profiling Report</title>' in report.to_html(), 'Profile report should be generated.'

def test_issue51_similar():
    if False:
        while True:
            i = 10
    df = pd.DataFrame({'test': ['', 'hoi', None], 'blest': [None, '', 'geert'], 'bert': ['snor', '', None]})
    report = ProfileReport(df, title='Pandas Profiling Report', progress_bar=False, explorative=True)
    report.config.vars.num.low_categorical_threshold = 0
    assert '<title>Pandas Profiling Report</title>' in report.to_html(), 'Profile report should be generated.'

def test_issue51_empty():
    if False:
        print('Hello World!')
    df = pd.DataFrame({'test': ['', '', '', '', ''], 'blest': ['', '', '', '', ''], 'bert': ['', '', '', '', '']})
    report = ProfileReport(df, title='Pandas Profiling Report', progress_bar=False, explorative=True)
    report.config.vars.num.low_categorical_threshold = 0
    assert 'cramers' not in report.get_description().correlations or (report.get_description().correlations['cramers'].values == np.ones((3, 3))).all()

def test_issue51_identical():
    if False:
        while True:
            i = 10
    df = pd.DataFrame({'test': ['v1', 'v1', 'v1'], 'blest': ['v1', 'v1', 'v1'], 'bert': ['v1', 'v1', 'v1']})
    report = ProfileReport(df, title='Pandas Profiling Report', progress_bar=False, explorative=True)
    report.config.vars.num.low_categorical_threshold = 0
    assert report.get_description().correlations == {}
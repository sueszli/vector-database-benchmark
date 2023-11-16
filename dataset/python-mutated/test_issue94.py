"""
Test for issue 94:
https://github.com/ydataai/ydata-profiling/issues/94

Test based on:
https://stackoverflow.com/questions/52926527/pandas-profiling-1-4-1-throws-zerodivisionerror-for-valid-data-set-which-pandas
"""
from pathlib import Path
import pandas as pd
import ydata_profiling

def test_issue94(tmpdir):
    if False:
        while True:
            i = 10
    file_path = Path(str(tmpdir)) / 'issue94.csv'
    file_path.write_text('CourseName\nPHY\nMATHS\nMATHS\nMATHS\nPHY\nPHY\nPHY\nCHEM\nCHEM\nCHEM')
    df = pd.read_csv(str(file_path), parse_dates=True)
    profile = ydata_profiling.ProfileReport(df, title='Pandas Profiling Report')
    assert '<title>Pandas Profiling Report</title>' in profile.to_html()
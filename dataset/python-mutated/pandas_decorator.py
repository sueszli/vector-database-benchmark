"""This file add the decorator on the DataFrame object."""
from pandas import DataFrame
from ydata_profiling.profile_report import ProfileReport

def profile_report(df: DataFrame, **kwargs) -> ProfileReport:
    if False:
        for i in range(10):
            print('nop')
    'Profile a DataFrame.\n\n    Args:\n        df: The DataFrame to profile.\n        **kwargs: Optional arguments for the ProfileReport object.\n\n    Returns:\n        A ProfileReport of the DataFrame.\n    '
    p = ProfileReport(df, **kwargs)
    return p
DataFrame.profile_report = profile_report
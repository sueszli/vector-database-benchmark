"""
Test for issue 353:
https://github.com/ydataai/ydata-profiling/issues/353
"""
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

def test_issue353():
    if False:
        return 10
    df = pd.DataFrame(np.random.rand(100, 5), columns=['a', 'b', 'c', 'd', 'e'])
    df['a'] = df['a'].multiply(5).astype('int').astype('category')
    profile = ProfileReport(df, title='Pandas Profiling Report', html={'style': {'full_width': True}}, progress_bar=False)
    assert len(profile.to_html()) > 0
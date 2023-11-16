import pandas as pd
from recommenders.datasets import criteo

def test_criteo_privacy(criteo_first_row):
    if False:
        while True:
            i = 10
    'Check that there are no privacy concerns. In Criteo, we check that the\n    data is anonymized.\n    '
    df = criteo.load_pandas_df(size='sample')
    assert df.loc[0].equals(pd.Series(criteo_first_row))
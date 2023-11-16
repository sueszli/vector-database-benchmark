import sys
if sys.version_info < (3, 0):
    from nose.plugins.skip import SkipTest
    raise SkipTest
from io import StringIO
import pandas as pd
from data_names import csv
from mlxtend.text import generalize_names, generalize_names_duplcheck

def test_generalize_names_duplcheck():
    if False:
        return 10
    df = pd.read_csv(StringIO(csv))
    dupl = any(df['Name'].apply(generalize_names).duplicated())
    assert dupl is True
    df_new = generalize_names_duplcheck(df=df, col_name='Name')
    no_dupl = any(df_new['Name'].duplicated())
    assert no_dupl is False
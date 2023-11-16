import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
import pandas as pd
import numpy as np

def difflag1():
    if False:
        i = 10
        return i + 15
    df = pd.DataFrame(np.random.randint(0, 100, size=(1000000, 1)), columns=list('A'))
    df_diff = df.diff()
    df_diff_h2o = h2o.H2OFrame(df_diff)
    fr = h2o.H2OFrame(df)
    fr_diff = fr.difflag1()
    diff = abs(df_diff_h2o[1:df_diff_h2o.nrow, :] - fr_diff[1:fr_diff.nrow, :])
    assert diff.max() < 1e-10, 'expected equal differencing'
if __name__ == '__main__':
    pyunit_utils.standalone_test(difflag1)
else:
    difflag1()
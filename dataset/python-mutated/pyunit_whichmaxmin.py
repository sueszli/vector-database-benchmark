import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
import pandas as pd

def whichmaxmin():
    if False:
        i = 10
        return i + 15
    f1 = h2o.create_frame(rows=10000, cols=100, categorical_fraction=0, missing_fraction=0, seed=1234)
    f2 = f1.as_data_frame(use_pandas=True)
    which_max_col = f1.idxmax()
    which_max_col = which_max_col.transpose()
    which_max_col_pd = f2.idxmax(axis=0)
    which_max_col_pd = h2o.H2OFrame(pd.DataFrame(which_max_col_pd, columns=['C1']))
    diff_max_col_idx = which_max_col - which_max_col_pd
    assert diff_max_col_idx.sum() == 0
    which_min_col = f1.idxmin()
    which_min_col = which_min_col.transpose()
    which_min_col_pd = f2.idxmin(axis=0)
    which_min_col_pd = h2o.H2OFrame(pd.DataFrame(which_min_col_pd, columns=['C1']))
    diff_min_col_idx = which_min_col - which_min_col_pd
    assert diff_min_col_idx.sum() == 0
    which_max_row = f1.idxmax(axis=1)
    which_max_row_pd = f2.idxmax(axis=1)
    which_max_row_pd = h2o.H2OFrame(pd.DataFrame(which_max_row_pd, columns=['C1']))
    which_max_row_pd = which_max_row_pd.ascharacter().lstrip('C').asnumeric() - 1
    diff_max_row_idx = which_max_row - which_max_row_pd
    assert diff_max_row_idx.sum() == 0
    which_min_row = f1.idxmin(axis=1)
    which_min_row_pd = f2.idxmin(axis=1)
    which_min_row_pd = h2o.H2OFrame(pd.DataFrame(which_min_row_pd, columns=['C1']))
    which_min_row_pd = which_min_row_pd.ascharacter().lstrip('C').asnumeric() - 1
    diff_min_row_idx = which_min_row - which_min_row_pd
    assert diff_min_row_idx.sum() == 0
if __name__ == '__main__':
    pyunit_utils.standalone_test(whichmaxmin)
else:
    whichmaxmin()
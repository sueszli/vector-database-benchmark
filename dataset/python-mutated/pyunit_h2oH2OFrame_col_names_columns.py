import sys, os
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def h2o_H2OFrame_col_names():
    if False:
        print('Hello World!')
    '\n    Python API test: h2o.frame.H2OFrame.col_names(), h2o.frame.H2OFrame.columns()\n\n    Copied from pyunit_colnames.py\n    '
    iris_wheader = h2o.import_file(pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    expected_names = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    assert iris_wheader.col_names == expected_names == iris_wheader.columns, 'Expected {0} for column names but got {1}'.format(expected_names, iris_wheader.col_names)
pyunit_utils.standalone_test(h2o_H2OFrame_col_names)
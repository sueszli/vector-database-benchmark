import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.utils.typechecks import assert_is_type
from h2o.frame import H2OFrame

def h2o_H2OFrame_isna():
    if False:
        for i in range(10):
            print('nop')
    '\n    Python API test: h2o.frame.H2OFrame.isna()\n\n    Copied from pyunit_isin.py\n    '
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    temp = iris.isna()
    assert_is_type(temp, H2OFrame)
    assert not temp.all(), 'h2o.H2OFrame.isna() command is not working.'
    iris_NA = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader_NA.csv'))
    temp_NA = iris_NA.isna()
    assert_is_type(temp_NA, H2OFrame)
    assert temp_NA.any(), 'h2o.H2OFrame.isna() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_isna)
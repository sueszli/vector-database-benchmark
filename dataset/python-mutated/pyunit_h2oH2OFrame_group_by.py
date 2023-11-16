import sys
sys.path.insert(1, '../../../')
import h2o
from h2o.group_by import GroupBy
from tests import pyunit_utils
from h2o.utils.typechecks import assert_is_type

def h2o_H2OFrame_group_by():
    if False:
        return 10
    '\n    Python API test: h2o.frame.H2OFrame.group_by(by)\n\n    Copied from pyunit_groupby.py\n    '
    h2o_iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    grouped = h2o_iris.group_by(['class'])
    assert_is_type(grouped, GroupBy)
    grouped = h2o_iris.group_by('class')
    assert_is_type(grouped, GroupBy)
    grouped = h2o_iris.group_by(4)
    assert_is_type(grouped, GroupBy)
    grouped = h2o_iris.group_by([4])
    assert_is_type(grouped, GroupBy)
pyunit_utils.standalone_test(h2o_H2OFrame_group_by)
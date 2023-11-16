import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o
from h2o.utils.typechecks import assert_is_type
from h2o.group_by import GroupBy
from h2o.frame import H2OFrame

def h2ogroup_by_GroupBy_AllOps():
    if False:
        return 10
    "\n    Python API test: h2o.group_by.GroupBy(fr, by), h2o.group_by.GroupBy.count(na='all'), h2o.group_by.GroupBy.frame,\n    h2o.group_by.GroupBy.count, h2o.group_by.GroupBy.mode, h2o.group_by.GroupBy.max, h2o.group_by.GroupBy.mean,\n    h2o.group_by.GroupBy.min, h2o.group_by.GroupBy.sd, h2o.group_by.GroupBy.ss, h2o.group_by.GroupBy.sum,\n    h2o.group_by.GroupBy.var\n\n    Copied from pyunit_groupby_allOps.py\n    "
    h2o_prostate = h2o.import_file(path=pyunit_utils.locate('smalldata/prostate/prostate_cat.csv'))
    by = ['CAPSULE', 'RACE']
    groupbyObj = GroupBy(fr=h2o_prostate, by=by)
    assert_is_type(groupbyObj, GroupBy)
    counts = groupbyObj.count(na='all')
    assert_is_type(counts, GroupBy)
    countsInfo = counts.get_frame()
    assert_is_type(countsInfo, H2OFrame)
    assert countsInfo.shape == (h2o_prostate['CAPSULE'].nlevels()[0] * h2o_prostate['RACE'].nlevels()[0], len(by) + 1), 'h2o.group_by.GroupBy.count() command is not working.'
    assert_is_type(counts.frame, H2OFrame)
    verifyOps(GroupBy(fr=h2o_prostate, by=by).mode(col=['DPROS'], na='rm'), (h2o_prostate['CAPSULE'].nlevels()[0] * h2o_prostate['RACE'].nlevels()[0], len(by) + 1), [2], [1.0], 'h2o.group_by.GroupBy.mode()')
    h2o_iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    groupbyObj = GroupBy(fr=h2o_iris, by='class')
    verifyOps(GroupBy(fr=h2o_iris, by='class').max(col=None, na='all'), (3, 5), ['max_sepal_len', 'max_sepal_wid', 'max_petal_wid'], [5.8, 3.4, 2.5], 'h2o.group_by.GroupBy.max()')
    verifyOps(GroupBy(fr=h2o_iris, by='class').mean(col=None, na='all'), (3, 5), ['mean_sepal_len', 'mean_sepal_wid', 'mean_petal_wid'], [5.006, 2.77, 2.026], 'h2o.group_by.GroupBy.mean()')
    verifyOps(GroupBy(fr=h2o_iris, by='class').min(col=None, na='all'), (3, 5), ['min_sepal_len', 'min_sepal_wid', 'min_petal_wid'], [4.3, 2.0, 1.4], 'h2o.group_by.GroupBy.min()')
    verifyOps(GroupBy(fr=h2o_iris, by='class').sd(col=None, na='all'), (3, 5), ['sdev_sepal_len', 'sdev_sepal_wid', 'sdev_petal_wid'], [0.352489687213, 0.313798323378, 0.274650055637], 'h2o.group_by.GroupBy.sd()')
    verifyOps(GroupBy(fr=h2o_iris, by='class').ss(col=None, na='all'), (3, 5), ['sumSquares_sepal_len', 'sumSquares_sepal_wid', 'sumSquares_petal_wid'], [1259.09, 388.47, 208.93], 'h2o.group_by.GroupBy.ss()')
    verifyOps(GroupBy(fr=h2o_iris, by='class').sum(col=None, na='all'), (3, 5), ['sum_sepal_len', 'sum_sepal_wid', 'sum_petal_wid'], [250.3, 138.5, 101.3], 'h2o.group_by.GroupBy.sum()')
    verifyOps(GroupBy(fr=h2o_iris, by='class').var(col=None, na='all'), (3, 5), ['var_sepal_len', 'var_sepal_wid', 'var_petal_wid'], [0.352489687213 * 0.352489687213, 0.313798323378 * 0.313798323378, 0.274650055637 * 0.274650055637], 'h2o.group_by.GroupBy.var()')

def verifyOps(opers, shapeS, threshold_name, threshold_val, groupByCommand):
    if False:
        for i in range(10):
            print('nop')
    assert_is_type(opers, GroupBy)
    operInfo = opers.get_frame()
    assert_is_type(operInfo, H2OFrame)
    assert operInfo.shape == shapeS, '{0} command is not working.'.format(groupByCommand)
    for index in range(len(threshold_val)):
        assert abs(operInfo[index, threshold_name[index]] - threshold_val[index]) < 1e-06, '{0} command is not working.'.format(groupByCommand)
pyunit_utils.standalone_test(h2ogroup_by_GroupBy_AllOps)
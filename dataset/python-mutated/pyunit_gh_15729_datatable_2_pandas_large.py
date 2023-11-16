import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from h2o.utils.shared_utils import can_use_pandas, can_use_datatable, can_install_datatable
import time

def test_frame_conversion(dataset, compareTime):
    if False:
        i = 10
        return i + 15
    print('Performing data conversion to pandas for dataset: {0}'.format(dataset))
    h2oFrame = h2o.import_file(pyunit_utils.locate(dataset))
    startT = time.time()
    original_pandas_frame = h2oFrame.as_data_frame()
    oldTime = time.time() - startT
    print('H2O frame to Pandas frame conversion time: {0}'.format(oldTime))
    startT = time.time()
    new_pandas_frame = h2oFrame.as_data_frame(multi_thread=True)
    newTime = time.time() - startT
    print('H2O frame to Pandas frame conversion time using datatable: {0}'.format(newTime))
    if compareTime:
        assert newTime <= oldTime, ' original frame conversion time: {0} should exceed new frame conversion time:{1} but is not.'.format(oldTime, newTime)
    new_types = new_pandas_frame.dtypes
    old_types = original_pandas_frame.dtypes
    ncol = h2oFrame.ncol
    for ind in range(ncol):
        assert new_types[ind] == old_types[ind], 'Expected column types: {0}, actual column types: {1}'.format(old_types[ind], new_types[ind])

def test_polars_pandas():
    if False:
        return 10
    if not can_install_datatable():
        print("Datatable doesn't run on Python 3.{0} for now.".format(sys.version_info.minor))
        return
    if not can_use_pandas():
        pyunit_utils.install('pandas')
    import pandas
    if not can_use_datatable():
        pyunit_utils.install('datatable')
    import datatable
    test_frame_conversion('bigdata/laptop/jira/PUBDEV_5266_merge_with_string_columns/PUBDEV_5266_f1.csv', False)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_polars_pandas)
else:
    test_polars_pandas()
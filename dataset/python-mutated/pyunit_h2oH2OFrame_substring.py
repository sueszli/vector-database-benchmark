import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o

def h2o_H2OFrame_substring():
    if False:
        while True:
            i = 10
    '\n    Python API test: h2o.frame.H2OFrame.substring(start_index, end_index=None)\n\n    Copied from pyunit_sub_gsub.py\n    '
    frame = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris.csv'))
    frame['C5'] = frame['C5'].substring(0, 5)
    assert (frame['C5'] == 'Iris-').sum() == frame.nrow, 'h2o.H2OFrame.substring() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_substring)
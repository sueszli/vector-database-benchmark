import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o
from h2o.utils.typechecks import assert_is_type
from h2o.frame import H2OFrame

def h2o_H2OFrame_cbind():
    if False:
        i = 10
        return i + 15
    '\n    Python API test: h2o.frame.H2OFrame.cbind(data)\n\n    Copied from pyunit_cbind.py\n    '
    hdf = h2o.import_file(path=pyunit_utils.locate('smalldata/jira/pub-180.csv'))
    (hdfrows, hdfcols) = hdf.shape
    hdf2 = hdf.cbind(hdf)
    assert_is_type(hdf2, H2OFrame)
    assert hdf2.shape == (hdfrows, 2 * hdfcols), 'h2o.H2OFrame.cbind() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_cbind)
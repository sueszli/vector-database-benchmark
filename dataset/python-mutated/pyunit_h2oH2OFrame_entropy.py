import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.utils.typechecks import assert_is_type
from h2o.frame import H2OFrame

def h2o_H2OFrame_entropy():
    if False:
        i = 10
        return i + 15
    '\n    Python API test: h2o.frame.H2OFrame.entropy()\n\n    copied from pyunit_entropy.py\n    '
    frame = h2o.H2OFrame.from_python(['redrum'])
    g = frame.entropy()
    assert_is_type(g, H2OFrame)
    assert abs(g[0, 0] - 2.25162916739) < 1e-06, 'h2o.H2OFrame.entropy() command is not working.'
    strings = h2o.H2OFrame.from_python([''], column_types=['string'])
    assert_is_type(strings, H2OFrame)
    assert strings.entropy()[0, 0] == 0, 'h2o.H2OFrame.entropy() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_entropy)
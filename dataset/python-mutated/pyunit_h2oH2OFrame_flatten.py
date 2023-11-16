import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def h2o_H2OFrame_flatten():
    if False:
        return 10
    '\n    Python API test: h2o.frame.H2OFrame.flatten()\n\n    copied from pyunit_entropy.py\n    '
    frame = h2o.H2OFrame.from_python(['redrum'])
    g = frame.flatten()
    assert g == 'redrum', 'h2o.H2OFrame.flatten() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_flatten)
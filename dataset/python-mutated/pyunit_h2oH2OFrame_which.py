import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.utils.typechecks import assert_is_type
import numpy as np
from h2o.frame import H2OFrame

def h2o_H2OFrame_which():
    if False:
        print('Hello World!')
    '\n    Python API test: h2o.frame.H2OFrame.which()\n    '
    python_lists = np.random.randint(1, 5, (100, 1))
    h2oframe = h2o.H2OFrame(python_obj=python_lists)
    newFrame = h2oframe.which()
    assert_is_type(newFrame, H2OFrame)
    assert newFrame[1:h2oframe.nrow, 0].all(), 'h2o.H2OFrame.which() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_which)
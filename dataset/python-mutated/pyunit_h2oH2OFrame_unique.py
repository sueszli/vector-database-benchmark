import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.utils.typechecks import assert_is_type
import numpy as np
from h2o.frame import H2OFrame

def h2o_H2OFrame_unique():
    if False:
        print('Hello World!')
    '\n    Python API test: h2o.frame.H2OFrame.unique()\n    '
    python_lists = np.random.randint(-5, 5, (100, 1))
    h2oframe = h2o.H2OFrame(python_obj=python_lists)
    newFrame = h2oframe.unique()
    allLevels = h2oframe.asfactor().levels()[0]
    assert_is_type(newFrame, H2OFrame)
    assert len(allLevels) == newFrame.nrow, 'h2o.H2OFrame.unique command is not working.'
    newFrame = newFrame.asfactor()
    for rowIndex in range(newFrame.nrow):
        assert newFrame[rowIndex, 0] in allLevels, 'h2o.H2OFrame.unique command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_unique)
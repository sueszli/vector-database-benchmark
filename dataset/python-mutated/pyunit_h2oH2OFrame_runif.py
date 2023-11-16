import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
import numpy as np

def h2o_H2OFrame_runif():
    if False:
        print('Hello World!')
    '\n    Python API test: h2o.frame.H2OFrame.runif(seed=None)\n    '
    python_lists = np.random.uniform(0, 1, 10000)
    h2oframe = h2o.H2OFrame(python_obj=python_lists)
    h2oRunif = h2oframe.runif(seed=None)
    assert abs(h2oframe.mean().flatten() - h2oRunif.mean()) < 0.01, 'h2o.H2OFrame.runif() command is not working.'
    assert abs(h2oframe.sd()[0] - h2oRunif.sd()[0]) < 0.01, 'h2o.H2OFrame.runif() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_runif)
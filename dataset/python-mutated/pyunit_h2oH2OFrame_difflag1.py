import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o
import numpy as np
from h2o.utils.typechecks import assert_is_type
from h2o.frame import H2OFrame

def h2o_H2OFrame_difflag1():
    if False:
        print('Hello World!')
    '\n    Python API test: h2o.frame.H2OFrame.difflag1()\n    '
    python_object = [list(range(10)), list(range(10))]
    foo = h2o.H2OFrame(python_obj=np.transpose(python_object))
    diffs = foo[0].difflag1()
    results = diffs == 1.0
    assert_is_type(diffs, H2OFrame)
    assert results.sum().flatten() == foo.nrow - 1, 'h2o.H2OFrame.difflag1() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_difflag1)
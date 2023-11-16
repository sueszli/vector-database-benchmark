import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.utils.typechecks import assert_is_type
from random import randrange
import numpy as np
from h2o.frame import H2OFrame

def h2o_H2OFrame_logical_negation():
    if False:
        for i in range(10):
            print('nop')
    '\n    Python API test: h2o.frame.H2OFrame.logical_negation()\n    '
    row_num = randrange(1, 10)
    col_num = randrange(1, 10)
    python_lists = np.zeros((row_num, col_num))
    h2oframe = h2o.H2OFrame(python_obj=python_lists)
    clist = h2oframe.logical_negation()
    assert_is_type(clist, H2OFrame)
    assert clist.all(), 'h2o.H2OFrame.logical_negation() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_logical_negation)
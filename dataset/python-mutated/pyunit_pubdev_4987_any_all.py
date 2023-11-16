import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def h2o_H2OFrame_all():
    if False:
        return 10
    '\n    Python API test: h2o.frame.H2OFrame.all(), h2o.frame.H2OFrame.any()\n    '
    python_lists = [[True, False], [False, True], [True, True], [True, 'NA']]
    h2oframe = h2o.H2OFrame(python_obj=python_lists, na_strings=['NA'])
    assert not h2oframe.all(), 'h2o.H2OFrame.all() command is not working.'
    assert h2oframe.any(), 'h2o.H2OFrame.any() command is not working.'
    h2o.remove(h2oframe)
    python_lists = [[True, True], [True, True], [True, True], [True, 'NA']]
    h2oframe = h2o.H2OFrame(python_obj=python_lists, na_strings=['NA'])
    assert h2oframe.all(), 'h2o.H2OFrame.all() command is not working.'
    assert h2oframe.any(), 'h2o.H2OFrame.any() command is not working.'
    h2o.remove(h2oframe)
    python_lists = [[False, False], [False, False], [False, False], [False, 'NA']]
    h2oframe = h2o.H2OFrame(python_obj=python_lists, na_strings=['NA'])
    assert not h2oframe.all(), 'h2o.H2OFrame.all() command is not working.'
    assert h2oframe.any(), 'h2o.H2OFrame.any() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_all)
import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o

def h2o_H2OFrame_structure():
    if False:
        i = 10
        return i + 15
    '\n    Python API test: h2o.frame.H2OFrame.structure()\n    '
    frame = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris.csv'))
    frame.structure()
pyunit_utils.standalone_test(h2o_H2OFrame_structure)
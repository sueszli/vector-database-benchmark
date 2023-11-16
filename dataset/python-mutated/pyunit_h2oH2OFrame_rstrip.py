import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def h2o_H2OFrame_rstrip():
    if False:
        for i in range(10):
            print('nop')
    "\n    Python API test: h2o.frame.H2OFrame.rstrip(set='')\n\n    Copied from runit_lstrip.R\n    "
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris.csv'))
    iris['C5'] = iris['C5'].rstrip('color')
    newNames = iris['C5'].levels()[0]
    newStrip = ['Iris-setosa', 'Iris-versi', 'Iris-virginica']
    assert newNames == newStrip, 'h2o.H2OFrame.rstrip() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_rstrip)
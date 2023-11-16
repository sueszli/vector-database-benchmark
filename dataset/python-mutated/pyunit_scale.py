import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def center_scale():
    if False:
        return 10
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris.csv'))[0:4]
    foo = iris.scale()
    foo = iris.scale(center=True, scale=False)
    foo = iris.scale(center=False, scale=True)
    foo = iris.scale(center=False, scale=False)
    foo = iris[0].scale()
    foo = iris[1].scale(center=True, scale=False)
    foo = iris[2].scale(center=False, scale=True)
    foo = iris[3].scale(center=False, scale=False)
if __name__ == '__main__':
    pyunit_utils.standalone_test(center_scale)
else:
    center_scale()
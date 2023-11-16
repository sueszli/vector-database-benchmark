from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
import numpy as np

def sdev():
    if False:
        while True:
            i = 10
    iris_h2o = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    iris_np = np.genfromtxt(pyunit_utils.locate('smalldata/iris/iris_wheader.csv'), delimiter=',', skip_header=1, usecols=(0, 1, 2, 3))
    sd_np = np.std(iris_np, axis=0, ddof=1)
    sd_h2o = iris_h2o.sd()
    print(sd_h2o)
    for i in range(4):
        assert abs(sd_np[i] - sd_h2o[i]) < 1e-10, 'expected standard deviations to be the same'
    print(iris_h2o[0:2].sd())
if __name__ == '__main__':
    pyunit_utils.standalone_test(sdev)
else:
    sdev()
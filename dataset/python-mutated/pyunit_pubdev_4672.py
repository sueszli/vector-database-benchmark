import h2o
from tests import pyunit_utils
import numpy as np

def test_4672():
    if False:
        i = 10
        return i + 15
    width = 4
    data = data_fixture(100, width)
    data_means = np.mean(data, axis=0)
    fr = h2o.H2OFrame.from_python(data.tolist(), column_names=list('ABCD'))
    h2o_means = fr.apply(lambda x: x.mean(skipna=False))
    assert all([abs(data_means[i] - h2o_means[0, i]) < 1e-12 for i in range(0, width)]), 'Numpy and H2O column means need to match'

def data_fixture(height=100, width=4):
    if False:
        print('Hello World!')
    return np.random.randn(height, width)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_4672)
else:
    test_4672()
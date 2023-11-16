from builtins import zip
from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def binop_star():
    if False:
        while True:
            i = 10
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    (rows, cols) = iris.dim
    iris.show()
    res = iris * 99
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    for (x, y) in zip([res[c].sum() for c in range(cols - 1)], [86773.5, 45351.9, 55816.2, 17800.2]):
        assert abs(x - y) < 1e-07, 'unexpected column sums.'
    res = 5 * iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    res = iris[0] * iris[1]
    res.show()
    assert abs(res.sum() - 2670.98) < 0.01, 'expected different column sum'
    res = iris[0] * iris[1] * iris[2] * iris[3]
    res.show()
    assert abs(res.sum() - 16560.42) < 0.01, 'expected different sum'
    res = iris * iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    res = iris[0:2] * iris[1:3]
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == 2, 'dimension mismatch'
if __name__ == '__main__':
    pyunit_utils.standalone_test(binop_star)
else:
    binop_star()
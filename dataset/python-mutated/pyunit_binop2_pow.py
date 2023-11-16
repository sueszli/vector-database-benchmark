from builtins import zip
from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def binop_pow():
    if False:
        while True:
            i = 10
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader_65_rows.csv'))
    (rows, cols) = iris.dim
    iris.show()
    res = 2 ** iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    for (x, y) in zip([sum([res[r, c] for r in range(rows)]) for c in range(cols - 1)], [2689.579, 659.6639, 439.1082, 97.49004]):
        assert abs(x - y) < 0.01, 'expected same values'
    res = 1.1 ** iris[2]
    res2 = res[32, :] ** res[10, :]
    assert abs(res2 - 1.179319) < 1e-05, 'expected same values'
    res = 2 ** iris[0]
    res2 = res[32, :] ** 3
    assert int(res2) - 49667 == 0, 'expected same values'
    res = iris[0] ** iris[1] * iris[2] ** iris[3]
    assert int(res.sum()) - 47242.98 < 0.01, 'expected same values'
    res = iris ** iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    res = iris[0:2] ** iris[1:3]
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == 2, 'dimension mismatch'
    res = iris ** 2
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    for (x, y) in zip([res[c].sum() for c in range(cols - 1)], [1800.33, 709.32, 382.69, 30.74]):
        assert abs(x - y) < 0.01, 'expected same values'
if __name__ == '__main__':
    pyunit_utils.standalone_test(binop_pow)
else:
    binop_pow()
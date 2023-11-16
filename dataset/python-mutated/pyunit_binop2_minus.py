from builtins import zip
from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def binop_minus():
    if False:
        return 10
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader_65_rows.csv'))
    (rows, cols) = iris.dim
    res = 2 - iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    for (x, y) in zip([sum([res[r, c] for r in range(rows)]) for c in range(cols - 1)], [-209.9, -82.6, -6.9, 97.8]):
        assert abs(x - y) < 0.1, 'expected same values'
    res = 2 - iris[0]
    res2 = 1.1 - res[21, :]
    assert abs(res2 - 4.2) < 0.1, 'expected same values'
    try:
        res = 1.2 - iris[2]
        res2 = res[21, :] - iris
        print(res2.dim)
        assert False, ' Expected Frame dimension mismatch error'
    except Exception:
        pass
    try:
        res = 1.2 - iris[2]
        res2 = res[21, :] - iris[1]
        res2.show()
        assert False, 'Expected Frame dimension mismatch error'
    except Exception:
        pass
    res = 1.1 - iris[2]
    res2 = res[21, :] - res[10, :]
    assert abs(res2 - 0) < 0.1, 'expected same values'
    res = 2 - iris[0]
    res2 = res[21, :] - 3
    assert abs(res2 - -6.1) < 0.1, 'expected same values'
    try:
        res = 1.2 - iris[2]
        res2 = iris[1] - res[21, :]
        res2.show()
        assert False, 'Expected Frame dimension mismatch error'
    except Exception:
        pass
    res = iris - iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    res = iris[0:2] - iris[1:3]
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == 2, 'dimension mismatch'
    try:
        res = 1.2 - iris[2]
        res2 = iris - res[21, :]
        res2.show()
        assert False, 'Expected Frame dimension mismatch error'
    except Exception:
        pass
    res = iris - 2
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    for (x, y) in zip([sum([res[r, c] for r in range(rows)]) for c in range(cols - 1)], [209.9, 82.6, 6.9, -97.8]):
        assert abs(x - y) < 0.1, 'expected same values'
if __name__ == '__main__':
    pyunit_utils.standalone_test(binop_minus)
else:
    binop_minus()
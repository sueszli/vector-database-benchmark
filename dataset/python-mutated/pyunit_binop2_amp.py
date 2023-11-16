from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def binop_amp():
    if False:
        return 10
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader_65_rows.csv'))
    (rows, cols) = iris.dim
    res = iris[0] & iris[1]
    assert res.sum() == 65.0, 'expected all True'
    res = iris[2] & iris[1]
    assert res.sum() == 65.0, 'expected all True'
    res = 1.2 + iris[2]
    res2 = iris[1, :] & res[7, :].flatten()
    res2.show()
    res = iris & iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    res = iris[0:2] & iris[1:3]
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == 2, 'dimension mismatch'
    res = iris & 0
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    for c in range(cols - 1):
        for r in range(rows):
            assert res[r, c] == 0.0, 'expected False'
if __name__ == '__main__':
    pyunit_utils.standalone_test(binop_amp)
else:
    binop_amp()
from builtins import zip
from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def binop_plus():
    if False:
        for i in range(10):
            print('nop')
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader_65_rows.csv'))
    (rows, cols) = iris.dim
    iris.show()
    res = 2 + iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    for (x, y) in zip([res[c].sum() for c in range(cols - 1)], [469.9, 342.6, 266.9, 162.2]):
        assert abs(x - y) < 0.1, 'expected same values'
    res = 2 + iris[0]
    res2 = 1.1 + res[21, :]
    assert abs(res2 - 8.2) < 0.1, 'expected same values'
    res = 1.1 + iris[2]
    res2 = res[21, :] + res[10, :]
    assert abs(res2 - 5.2) < 0.1, 'expected same values'
    res = 2 + iris[0]
    res2 = res[21, :] + 3
    assert abs(res2 - 10.1) < 0.1, 'expected same values'
    res = iris + iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    res = iris[0:2] + iris[1:3]
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == 2, 'dimension mismatch'
    res = iris + 2
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    for (x, y) in zip([res[c].sum() for c in range(cols - 1)], [469.9, 342.6, 266.9, 162.2]):
        assert abs(x - y) < 0.1, 'expected same values'
if __name__ == '__main__':
    pyunit_utils.standalone_test(binop_plus)
else:
    binop_plus()
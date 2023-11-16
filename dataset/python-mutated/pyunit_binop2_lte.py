import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def binop_lte():
    if False:
        while True:
            i = 10
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    (rows, cols) = iris.dim
    iris.show()
    res = iris <= 5
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    new_rows = iris[res[0]].nrow
    assert new_rows == 32, 'wrong number of rows returned'
    res = 5 >= iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    new_rows = iris[res[0]].nrow
    assert new_rows == 32, 'wrong number of rows returned'
    res = iris[0] <= iris[1]
    res_rows = res.nrow
    assert res_rows == rows, 'dimension mismatch'
    new_rows = iris[res].nrow
    assert new_rows == 0, 'wrong number of rows returned'
    res = iris <= iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    res = iris[0:2] <= iris[1:3]
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == 2, 'dimension mismatch'
if __name__ == '__main__':
    pyunit_utils.standalone_test(binop_lte)
else:
    binop_lte()
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def binop_pipe():
    if False:
        return 10
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    (rows, cols) = iris.dim
    iris.show()
    res = 5 | iris
    (rows, cols) = res.dim
    assert rows == rows and cols == cols, 'dimension mismatch'
    res = iris | 1
    (rows, cols) = res.dim
    assert rows == rows and cols == cols, 'dimension mismatch'
    res = iris[0] | iris[1]
    rows = len(res)
    assert rows == rows, 'dimension mismatch'
    res = iris[0] | 1
    rows = res.nrow
    assert rows == rows, 'dimension mismatch'
    new_rows = iris[res].nrow
    assert new_rows == rows, 'wrong number of rows returned'
    res = 1 | iris[1]
    rows = res.nrow
    assert rows == rows, 'dimension mismatch'
    new_rows = iris[res].nrow
    assert new_rows == rows, 'wrong number of rows returned'
    res = iris | iris
    (rows, cols) = res.dim
    assert rows == rows and cols == cols, 'dimension mismatch'
    res = iris[0:2] | iris[1:3]
    (rows, cols) = res.dim
    assert rows == rows and cols == 2, 'dimension mismatch'
if __name__ == '__main__':
    pyunit_utils.standalone_test(binop_pipe)
else:
    binop_pipe()
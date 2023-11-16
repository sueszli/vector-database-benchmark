from builtins import zip
from builtins import range
from past.utils import old_div
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def binop_div():
    if False:
        print('Hello World!')
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    (rows, cols) = iris.dim
    iris.show()
    res = iris / 5
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    for (x, y) in zip([res[c].sum() for c in range(cols - 1)], [175.3, 91.62, 112.76, 35.96]):
        assert abs(x - y) < 1e-07, 'unexpected column sums.'
    res = 5 / iris
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    res = old_div(iris[0], iris[1])
    res.show()
    assert abs(res.sum() - 293.2717) < 0.0001, 'expected different column sum'
    res = iris[0] / iris[1] / iris[2] / iris[3]
    res.show()
    assert abs(res.sum() - 283.977) < 0.0001, 'expected different sum'
    res = old_div(iris[0], 5)
    res.show()
    assert abs(res.sum() - 175.3) < 1e-07, 'expected different column sum'
    res = old_div(iris, iris)
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == cols, 'dimension mismatch'
    res = old_div(iris[0:2], iris[1:3])
    (res_rows, res_cols) = res.dim
    assert res_rows == rows and res_cols == 2, 'dimension mismatch'
if __name__ == '__main__':
    pyunit_utils.standalone_test(binop_div)
else:
    binop_div()
import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def countmatches_check():
    if False:
        while True:
            i = 10
    frame = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris.csv'), col_types=['numeric', 'numeric', 'numeric', 'numeric', 'string'])
    result = frame['C5'].countmatches('o')
    assert result.nrow == 150 and result.ncol == 1
    assert result[0, 0] == 1 and result[50, 0] == 2 and (result[100, 0] == 0), 'Expected 1, 2, 0 but got {0}, {1}, and {2}'.format(result[0, 0], result[50, 0], result[100, 0])
    vec = frame['C5']
    result = vec.countmatches('ic')
    assert result.nrow == 150 and result.ncol == 1
    assert result[0, 0] == 0 and result[50, 0] == 1 and (result[100, 0] == 1), 'Expected 0, 1, 1 but got {0}, {1}, and {2}'.format(result[0, 0], result[50, 0], result[100, 0])
    vec = frame['C5']
    result = vec.countmatches(['ic', 'ri', 'ca'])
    assert result.nrow == 150 and result.ncol == 1
    assert result[0, 0] == 1 and result[50, 0] == 2 and (result[100, 0] == 3), 'Expected 1, 2, 3 but got {0}, {1}, and {2}'.format(result[0, 0], result[50, 0], result[100, 0])
if __name__ == '__main__':
    pyunit_utils.standalone_test(countmatches_check)
else:
    countmatches_check()
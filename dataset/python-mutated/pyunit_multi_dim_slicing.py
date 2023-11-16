import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def multi_dim_slicing():
    if False:
        print('Hello World!')
    prostate = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    pros = prostate[47:51, 7]
    assert pros[0, 0] == 16.3, 'Incorrect slicing result'
    pros = prostate[172, 8]
    assert pros == 7, 'Incorrect slicing result'
    pros = prostate[170:176, 2]
    assert pros[0, 0] == 74, 'Incorrect slicing result'
    assert pros[1, 0] == 71, 'Incorrect slicing result'
    assert pros[2, 0] == 60, 'Incorrect slicing result'
    assert pros[3, 0] == 62, 'Incorrect slicing result'
    assert pros[4, 0] == 71, 'Incorrect slicing result'
    assert pros[5, 0] == 67, 'Incorrect slicing result'
    pros = prostate[188, 0:3]
    assert pros[0, 0] == 189, 'Incorrect slicing result'
    assert pros[0, 1] + 1 == 2, 'Incorrect slicing result'
    assert pros[0, 2] == 69, 'Incorrect slicing result'
    pros = prostate[83:86, 1:4]
    assert pros[0, 0] == 0, 'Incorrect slicing result'
    assert pros[0, 1] == 75, 'Incorrect slicing result'
    assert pros[0, 2] - 1 == 0, 'Incorrect slicing result'
    assert pros[1, 0] == 0, 'Incorrect slicing result'
    assert pros[1, 1] + 75 == 150, 'Incorrect slicing result'
    assert pros[1, 2] == 1, 'Incorrect slicing result'
    assert pros[2, 0] + 1 == 2, 'Incorrect slicing result'
    assert pros[2, 1] == 75, 'Incorrect slicing result'
    assert pros[2, 2] == 1, 'Incorrect slicing result'
if __name__ == '__main__':
    pyunit_utils.standalone_test(multi_dim_slicing)
else:
    multi_dim_slicing()
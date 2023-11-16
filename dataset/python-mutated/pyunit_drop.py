import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def frame_checker(frame):
    if False:
        return 10
    assert frame.ncol == len(frame.names) == len(frame.types)
    assert set(frame.names) == set(frame.types)

def pyunit_drop():
    if False:
        for i in range(10):
            print('nop')
    pros = h2o.import_file(pyunit_utils.locate('smalldata/prostate/prostate.csv'))
    nc = pros.ncol
    nr = pros.nrow
    dropped_col_int = pros.drop(0)
    frame_checker(dropped_col_int)
    dropped_col_string = pros.drop('ID')
    frame_checker(dropped_col_string)
    dropped_col_int_array = pros.drop([0, 1])
    frame_checker(dropped_col_int_array)
    to_drop = ['ID', 'CAPSULE']
    dropped_col_string_array = pros.drop(to_drop)
    frame_checker(dropped_col_string_array)
    assert to_drop == ['ID', 'CAPSULE']
    dropped_row_array_0 = pros.drop([0], axis=0)
    dropped_row_array_1 = pros.drop([0, 1], axis=0)
    dropped_row_array_2 = pros.drop([0, 1, 2], axis=0)
    dropped_row_array_380 = pros.drop([379], axis=0)
    dropped_row_array_378 = pros.drop([378, 379], axis=0)
    dropped_row_array_377 = pros.drop([377, 378, 379], axis=0)
    assert dropped_col_int.ncol == nc - 1
    assert dropped_col_string.ncol == nc - 1
    assert dropped_col_int_array.ncol == nc - 2
    assert dropped_col_string_array.ncol == nc - 2
    assert dropped_col_int.names == pros.names[1:]
    assert dropped_col_string.names == pros.names[1:]
    assert dropped_col_int_array.names == pros.names[2:]
    assert dropped_col_string_array.names == pros.names[2:]
    assert dropped_col_int.types == pros[1:].types
    assert dropped_col_string.types == pros[1:].types
    assert dropped_col_int_array.types == pros[2:].types
    assert dropped_col_string_array.types == pros[2:].types
    assert dropped_row_array_0.nrow == nr - 1
    assert dropped_row_array_1.nrow == nr - 2
    assert dropped_row_array_2.nrow == nr - 3
    assert dropped_row_array_380.nrow == nr - 1
    assert dropped_row_array_378.nrow == nr - 2
    assert dropped_row_array_377.nrow == nr - 3
    assert pros.ncol == nc
    assert pros.nrow == nr
if __name__ == '__main__':
    pyunit_utils.standalone_test(pyunit_drop)
else:
    pyunit_drop()
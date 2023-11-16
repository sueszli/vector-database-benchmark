import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def import_folder():
    if False:
        for i in range(10):
            print('nop')
    tol_time = 200
    tol_numeric = 1e-05
    numElements2Compare = 0
    multi_file_csv1 = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/orc/synthetic_perfect_seperation_csv/balunbal.csv'))
    multi_file_csv2 = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/orc/synthetic_perfect_seperation_csv/unbalbal.csv'))
    multi_file_orc = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/orc/synthetic_perfect_separation'))
    try:
        assert pyunit_utils.compare_frames(multi_file_orc, multi_file_csv1, numElements2Compare, tol_time, tol_numeric, True), 'H2O frame parsed from multiple orc and single orc files are different!'
    except:
        assert pyunit_utils.compare_frames(multi_file_orc, multi_file_csv2, numElements2Compare, tol_time, tol_numeric, True), 'H2O frame parsed from multiple orc and single orc files are different!'
if __name__ == '__main__':
    pyunit_utils.standalone_test(import_folder)
else:
    import_folder()
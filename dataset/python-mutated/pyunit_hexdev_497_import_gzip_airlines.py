import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def import_folder():
    if False:
        print('Hello World!')
    tol_time = 200
    tol_numeric = 1e-05
    numElements2Compare = 0
    multi_file_gzip_comp = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/hexdev_497/airlines_small_csv.zip'))
    multi_file_csv = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/hexdev_497/airlines_small_csv/all_airlines.csv'))
    assert pyunit_utils.compare_frames(multi_file_csv, multi_file_gzip_comp, numElements2Compare, tol_time, tol_numeric, True), 'H2O frame parsed from zip directory and unzipped directory are different!'
if __name__ == '__main__':
    pyunit_utils.standalone_test(import_folder)
else:
    import_folder()
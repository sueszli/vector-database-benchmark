import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def import_folder():
    if False:
        i = 10
        return i + 15
    tol_time = 200
    tol_numeric = 1e-05
    numElements2Compare = 100
    multi_file_gzip_comp = h2o.import_file(path=pyunit_utils.locate('bigdata/laptop/parser/hexdev_497/milsongs_csv.zip'))
    multi_file_csv = h2o.import_file(path=pyunit_utils.locate('bigdata/laptop/parser/hexdev_497/milsongs_csv_gzip'))
    try:
        assert pyunit_utils.compare_frames(multi_file_csv, multi_file_gzip_comp, numElements2Compare, tol_time, tol_numeric, True), 'H2O frame parsed from multiple orc and single orc files are different!'
    except:
        multi_file_gzip_comp.summary()
        zip_summary = h2o.frame(multi_file_gzip_comp.frame_id)['frames'][0]['columns']
        multi_file_csv.summary()
        csv_summary = h2o.frame(multi_file_csv.frame_id)['frames'][0]['columns']
        pyunit_utils.compare_frame_summary(zip_summary, csv_summary)
if __name__ == '__main__':
    pyunit_utils.standalone_test(import_folder)
else:
    import_folder()
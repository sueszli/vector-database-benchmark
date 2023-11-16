import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def import_folder():
    if False:
        print('Hello World!')
    multi_file_csv = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/hexdev_497/airlines_first_header'))
    multi_file_gzip_comp = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/hexdev_497/airlines_first_header.zip'))
    multi_file_gzip_comp.summary()
    zip_summary = h2o.frame(multi_file_gzip_comp.frame_id)['frames'][0]['columns']
    multi_file_csv.summary()
    csv_summary = h2o.frame(multi_file_csv.frame_id)['frames'][0]['columns']
    pyunit_utils.compare_frame_summary(zip_summary, csv_summary)
if __name__ == '__main__':
    pyunit_utils.standalone_test(import_folder)
else:
    import_folder()
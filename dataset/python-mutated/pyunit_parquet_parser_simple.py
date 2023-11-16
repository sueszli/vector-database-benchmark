import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def parquet_parse_simple():
    if False:
        i = 10
        return i + 15
    '\n    Tests Parquet parser by comparing the summary of the original csv frame with the h2o parsed Parquet frame.\n    Basic use case of importing files with auto-detection of column types.\n    :return: None if passed.  Otherwise, an exception will be thrown.\n    '
    csv = h2o.import_file(path=pyunit_utils.locate('smalldata/airlines/AirlinesTrain.csv.zip'))
    parquet = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/parquet/airlines-simple.snappy.parquet'))
    csv.summary()
    csv_summary = h2o.frame(csv.frame_id)['frames'][0]['columns']
    parquet.summary()
    parquet_summary = h2o.frame(parquet.frame_id)['frames'][0]['columns']
    pyunit_utils.compare_frame_summary(csv_summary, parquet_summary)
if __name__ == '__main__':
    pyunit_utils.standalone_test(parquet_parse_simple)
else:
    parquet_parse_simple()
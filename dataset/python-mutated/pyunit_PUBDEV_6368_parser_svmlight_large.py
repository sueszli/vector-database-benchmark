import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def test_parser_svmlight_column_skip():
    if False:
        while True:
            i = 10
    f1 = h2o.import_file(pyunit_utils.locate('bigdata/laptop/parser/100k.svm'))
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_parser_svmlight_column_skip)
else:
    test_parser_svmlight_column_skip()
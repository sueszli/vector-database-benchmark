from builtins import str
import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from random import randint
'\nThis test takes all orc files collected by Tom K and try to parse them into H2O frames.\nDue to test duration limitation, we do not parse all the files.  Instead, we randomly\nchoose about 30% of the files and parse them into H2O frames.  If all pareses are successful,\nthe test pass and else it fails.\n'

def orc_parser_test():
    if False:
        for i in range(10):
            print('nop')
    allOrcFiles = ['smalldata/parser/orc/TestOrcFile.columnProjection.orc', 'smalldata/parser/orc/bigint_single_col.orc', 'smalldata/parser/orc/TestOrcFile.emptyFile.orc', 'smalldata/parser/orc/bool_single_col.orc', 'smalldata/parser/orc/demo-11-zlib.orc', 'smalldata/parser/orc/TestOrcFile.testDate1900.orc', 'smalldata/parser/orc/demo-12-zlib.orc', 'smalldata/parser/orc/TestOrcFile.testDate2038.orc', 'smalldata/parser/orc/double_single_col.orc', 'smalldata/parser/orc/TestOrcFile.testMemoryManagementV11.orc', 'smalldata/parser/orc/float_single_col.orc', 'smalldata/parser/orc/TestOrcFile.testMemoryManagementV12.orc', 'smalldata/parser/orc/int_single_col.orc', 'smalldata/parser/orc/TestOrcFile.testPredicatePushdown.orc', 'smalldata/parser/orc/nulls-at-end-snappy.orc', 'smalldata/parser/orc/TestOrcFile.testSnappy.orc', 'smalldata/parser/orc/orc_split_elim.orc', 'smalldata/parser/orc/TestOrcFile.testStringAndBinaryStatistics.orc', 'smalldata/parser/orc/TestOrcFile.testStripeLevelStats.orc', 'smalldata/parser/orc/smallint_single_col.orc', 'smalldata/parser/orc/string_single_col.orc', 'smalldata/parser/orc/tinyint_single_col.orc', 'smalldata/parser/orc/TestOrcFile.testWithoutIndex.orc']
    for fIndex in range(len(allOrcFiles)):
        tab_test = h2o.import_file(path=pyunit_utils.locate(allOrcFiles[fIndex]))
if __name__ == '__main__':
    pyunit_utils.standalone_test(orc_parser_test)
else:
    orc_parser_test()
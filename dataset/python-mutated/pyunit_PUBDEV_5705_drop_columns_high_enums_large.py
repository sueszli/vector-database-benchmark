import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
import os

def test_column_skip_high_cardinality():
    if False:
        for i in range(10):
            print('nop')
    tmpdir = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath('__file__')), '..', 'results'))
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    savefilenamewithpath = os.path.join(tmpdir, 'in.csv')
    fwriteFile = open(savefilenamewithpath, 'w')
    nrow = 10000000
    for rowindex in range(nrow):
        writeWords = 'a' + str(rowindex) + ',' + str(rowindex) + '\n'
        fwriteFile.write(writeWords)
    fwriteFile.close()
    try:
        h2o.upload_file(savefilenamewithpath, col_types=['enum', 'int'])
        assert False, 'Test should have thrown an exception due to all columns are skipped'
    except Exception as ex:
        print(ex)
        h2o.upload_file(savefilenamewithpath, col_types=['int'], skipped_columns=[0])
        print('Test passed! Parsed with large enum columns skipped!')
        pass
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_column_skip_high_cardinality)
else:
    test_column_skip_high_cardinality()
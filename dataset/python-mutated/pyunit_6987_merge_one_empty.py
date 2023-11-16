import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def mergeOneEmptyFrame():
    if False:
        print('Hello World!')
    file1 = h2o.H2OFrame({'A1': [1], 'A2': [0]})
    file2 = h2o.H2OFrame({'A1': [], 'A2': []})
    f1Mergef2 = file1.merge(file2, all_x=True)
    f2Mergef1 = file2.merge(file1, all_y=True)
    print('checking merge empty with all_y = True.  row 0 col 0: {0}, row 0 col 1: {1}'.format(f2Mergef1[0, 'A1'], f2Mergef1[0, 'A2']))
    print('checking merge empty with all_x = True. row 0 col 0: {0}, row 0 col 1: {1}'.format(f1Mergef2[0, 'A1'], f1Mergef2[0, 'A2']))
    assert f2Mergef1[0, 'A1'] == 1, 'f2Mergef1: Expected content 1 at row 0, col 0 but actual content is {0}'.format(f2Mergef1[0, 'A1'])
    assert f2Mergef1[0, 'A2'] == 0, 'f2Mergef1: Expected content 0 at row 0, col 1 but actual content is {0}'.format(f2Mergef1[0, 'A2'])
    assert f1Mergef2[0, 'A1'] == 1, 'f1Mergef2: Expected content 1 at row 0, col 0 but actual content is {0}'.format(f1Mergef2[0, 'A1'])
    assert f1Mergef2[0, 'A2'] == 0, 'f1Mergef2: Expected content 0 at row 0, col 1 but actual content is {0}'.format(f1Mergef2[0, 'A2'])
    assert f1Mergef2.nrow == 1, 'Expected one row  but actual number of row is {0}!'.format(f1Mergef2.nrows)
    assert f2Mergef1.nrow == 1, 'Expected one row  but actual number of row is {0}!'.format(f2Mergef1.nrows)
    assert f1Mergef2.ncols == 2, 'Expected two columns but actual number of row is {0}!'.format(f1Mergef2.ncols)
    assert f2Mergef1.ncols == 2, 'Expected two columns but actual number of row is {0}!'.format(f2Mergef1.ncols)
    f1Mergef2 = file1.merge(file2)
    f2Mergef1 = file2.merge(file1)
    f2Mergef2 = file2.merge(file2)
    assert f1Mergef2.nrows == 0, 'Expected empty rows but actual number of row is {0}!'.format(f1Mergef2.nrows)
    assert f2Mergef1.nrows == 0, 'Expected empty rows but actual number of row is {0}!'.format(f2Mergef1.nrows)
    assert f2Mergef2.nrows == 0, 'Expected empty rows but actual number of row is {0}!'.format(f2Mergef2.nrows)
if __name__ == '__main__':
    pyunit_utils.standalone_test(mergeOneEmptyFrame)
else:
    mergeOneEmptyFrame()
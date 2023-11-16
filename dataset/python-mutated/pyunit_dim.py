import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
import numpy as np

def dim_checks():
    if False:
        while True:
            i = 10
    h2o_data = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    np_data = np.loadtxt(pyunit_utils.locate('smalldata/logreg/prostate.csv'), delimiter=',', skiprows=1)
    (h2o_rows, h2o_cols) = h2o_data.dim
    (np_rows, np_cols) = list(np_data.shape)
    print('The dimensions of h2o frame is: {0} x {1}'.format(h2o_rows, h2o_cols))
    print('The dimensions of numpy array is: {0} x {1}'.format(np_rows, np_cols))
    assert [h2o_rows, h2o_cols] == [np_rows, np_cols], 'expected equal number of columns and rows'
    h2o_slice = h2o_data[4]
    np_slice = np_data[:, 4]
    (h2o_rows, h2o_cols) = h2o_slice.dim
    np_rows = np_slice.shape[0]
    print('The dimensions of h2o column slice is: {0} x {1}'.format(h2o_rows, h2o_cols))
    print('The dimensions of numpy array column slice is: {0} x 1'.format(np_rows))
    assert [h2o_rows, h2o_cols] == [np_rows, 1], 'expected equal number of columns and rows'
    h2oColAmpFive = h2o_slice & 5
    assert h2oColAmpFive.nrow == h2o_rows, 'expected the number of rows to remain unchanged'
if __name__ == '__main__':
    pyunit_utils.standalone_test(dim_checks)
else:
    dim_checks()
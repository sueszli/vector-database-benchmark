from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
import random

def slicing_shape():
    if False:
        for i in range(10):
            print('nop')
    prostate = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    (rows, cols) = prostate.dim
    for ncols in range(1, cols + 1):
        (r, c) = prostate[0:ncols].dim
        assert r == rows, 'incorrect number of rows. correct: {0}, computed: {1}'.format(rows, r)
        assert c == ncols, 'incorrect number of cols. correct: {0}, computed: {1}'.format(ncols, c)
    for ncols in range(1, cols + 1):
        (r, c) = prostate[random.randint(0, rows - 1), 0:ncols].dim
        assert r == 1, 'incorrect number of rows. correct: {0}, computed: {1}'.format(1, r)
        assert c == ncols, 'incorrect number of cols. correct: {0}, computed: {1}'.format(ncols, c)
    for nrows in range(1, 10):
        (r, c) = prostate[0:nrows, random.randint(0, cols - 1)].dim
        assert r == nrows, 'incorrect number of rows. correct: {0}, computed: {1}'.format(nrows, r)
        assert c == 1, 'incorrect number of cols. correct: {0}, computed: {1}'.format(1, c)
    for nrows in range(1, 10):
        for ncols in range(1, cols + 1):
            (r, c) = prostate[0:nrows, 0:ncols].dim
            assert r == nrows, 'incorrect number of rows. correct: {0}, computed: {1}'.format(nrows, r)
            assert c == ncols, 'incorrect number of cols. correct: {0}, computed: {1}'.format(ncols, c)
if __name__ == '__main__':
    pyunit_utils.standalone_test(slicing_shape)
else:
    slicing_shape()
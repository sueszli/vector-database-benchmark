import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
import os
import random

def test_csv_parser_column_skip():
    if False:
        for i in range(10):
            print('nop')
    nrow = 10000
    ncol = 20
    seed = 12345
    frac1 = 0.16
    frac2 = 0.2
    f1 = h2o.create_frame(rows=nrow, cols=ncol, real_fraction=frac1, categorical_fraction=frac1, integer_fraction=frac1, binary_fraction=frac1, time_fraction=frac1, string_fraction=frac2, missing_fraction=0.1, has_response=False, seed=seed)
    tmpdir = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath('__file__')), '..', 'results'))
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    savefilenamewithpath = os.path.join(tmpdir, 'in.csv')
    h2o.download_csv(f1, savefilenamewithpath)
    skip_all = list(range(f1.ncol))
    skip_start_end = [0, f1.ncol - 1]
    skip_except_last = list(range(0, f1.ncol - 2))
    skip_except_first = list(range(1, f1.ncol))
    temp = list(range(0, f1.ncol))
    random.shuffle(temp)
    skip_random = []
    for index in range(0, f1.ncol // 2):
        skip_random.append(temp[index])
    skip_random.sort()
    try:
        importFileSkipAll = h2o.import_file(savefilenamewithpath, skipped_columns=skip_all)
        assert False, 'Test should have thrown an exception due to all columns are skipped'
    except:
        pass
    pyunit_utils.checkCorrectSkips(f1, savefilenamewithpath, skip_start_end)
    pyunit_utils.checkCorrectSkips(f1, savefilenamewithpath, skip_except_last)
    pyunit_utils.checkCorrectSkips(f1, savefilenamewithpath, skip_except_first)
    pyunit_utils.checkCorrectSkips(f1, savefilenamewithpath, skip_random)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_csv_parser_column_skip)
else:
    test_csv_parser_column_skip()
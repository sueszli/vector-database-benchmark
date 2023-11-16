import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
import random

def import_folder():
    if False:
        i = 10
        return i + 15
    '\n    This test will build a H2O frame from importing the bigdata/laptop/parser/orc/airlines_05p_orc_csv\n    from and build another H2O frame from the multi-file orc parser using multiple orc files that are\n    saved in the directory bigdata/laptop/parser/orc/airlines_05p_orc.  It will compare the two frames\n    to make sure they are equal.\n    :return: None if passed.  Otherwise, an exception will be thrown.\n    '
    csv = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/csv2orc/prostate_NA.csv'), na_strings=['\\N'])
    multi_file_orc1 = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/orc/prostate_NA.orc'))
    pyunit_utils.compare_frames_local(csv, multi_file_orc1, prob=0.01)
    path = pyunit_utils.locate('smalldata/parser/orc/prostate_NA.orc')
    skip_all = list(range(csv.ncol))
    skip_even = list(range(0, csv.ncol, 2))
    skip_odd = list(range(1, csv.ncol, 2))
    skip_start_end = [0, csv.ncol - 1]
    skip_except_last = list(range(0, csv.ncol - 2))
    skip_except_first = list(range(1, csv.ncol))
    temp = list(range(0, csv.ncol))
    random.shuffle(temp)
    skip_random = []
    for index in range(0, csv.ncol // 2):
        skip_random.append(temp[index])
    skip_random.sort()
    try:
        loadFileSkipAll = h2o.upload_file(path, skipped_columns=skip_all)
        sys.exit(1)
    except Exception as ex:
        print(ex)
        pass
    try:
        importFileSkipAll = h2o.import_file(path, skipped_columns=skip_all)
        sys.exit(1)
    except Exception as ex:
        print(ex)
        pass
    pyunit_utils.checkCorrectSkips(csv, path, skip_even)
    pyunit_utils.checkCorrectSkips(csv, path, skip_odd)
    pyunit_utils.checkCorrectSkips(csv, path, skip_start_end)
    pyunit_utils.checkCorrectSkips(csv, path, skip_except_last)
    pyunit_utils.checkCorrectSkips(csv, path, skip_except_first)
    pyunit_utils.checkCorrectSkips(csv, path, skip_random)
if __name__ == '__main__':
    pyunit_utils.standalone_test(import_folder)
else:
    import_folder()
from builtins import str
import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def orc_parser_timestamp_date():
    if False:
        return 10
    '\n    To verify that the orc parser is parsing correctly, we want to take a file we know (prostate_NA.csv), convert\n    it to an Orc file (prostate_NA.orc) and build two H2O frames out of them.   We compare them and verified that\n    they are the same.\n\n    Nidhi did this manually in Hive and verified that the parsing is correct.  I am automating the test here.\n\n    :return: None\n    '
    tol_time = 200
    tol_numeric = 1e-05
    numElements2Compare = 10
    h2oOrc = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/orc/prostate_NA.orc'))
    h2oCsv = h2o.import_file(path=pyunit_utils.locate('smalldata/parser/csv2orc/prostate_NA.csv'))
    assert pyunit_utils.compare_frames(h2oOrc, h2oCsv, numElements2Compare, tol_time, tol_numeric), 'H2O frame parsed from orc and csv files are different!'
if __name__ == '__main__':
    pyunit_utils.standalone_test(orc_parser_timestamp_date)
else:
    orc_parser_timestamp_date()
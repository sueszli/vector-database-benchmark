from builtins import str
import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def separator_test():
    if False:
        i = 10
        return i + 15
    'Test parsing of csv files having different separator characters.'
    path_tab = 'smalldata/parser/tabs.tsv'
    tab_test = h2o.import_file(path=pyunit_utils.locate(path_tab), destination_frame='tab_hex', sep='\t')
    assert tab_test.nrow == 3, 'Error: Number of rows are not correct.{0}'.format(str(tab_test.nrow))
    assert tab_test.ncol == 3, 'Error: Number of columns are not correct.{0}'.format(str(tab_test.ncol))
    tab_test_noarg = h2o.import_file(path=pyunit_utils.locate(path_tab), destination_frame='tab_hex')
    assert tab_test_noarg.nrow == 3, 'Error: Number of rows are not correct.{0}'.format(str(tab_test_noarg.nrow))
    assert tab_test_noarg.ncol == 3, 'Error: Number of columns are not correct.{0}'.format(str(tab_test_noarg.ncol))
    path_pipe = 'smalldata/parser/pipes.psv'
    pipe_test = h2o.import_file(path=pyunit_utils.locate(path_pipe), destination_frame='pipe_hex', sep='|')
    assert pipe_test.nrow == 3, 'Error: Number of rows are not correct.{0}'.format(str(pipe_test.nrow))
    assert pipe_test.ncol == 3, 'Error: Number of columns are not correct.{0}'.format(str(pipe_test.ncol))
    pipe_test_noarg = h2o.import_file(path=pyunit_utils.locate(path_pipe), destination_frame='pipe_hex')
    assert pipe_test_noarg.nrow == 3, 'Error: Number of rows are not correct.{0}'.format(str(pipe_test_noarg.nrow))
    assert pipe_test_noarg.ncol == 3, 'Error: Number of columns are not correct.{0}'.format(str(pipe_test_noarg.ncol))
    path_hive = 'smalldata/parser/test.hive'
    hive_test = h2o.import_file(path=pyunit_utils.locate(path_hive), destination_frame='hive_hex', sep='\x01')
    assert hive_test.nrow == 3, 'Error: Number of rows are not correct.{0}'.format(str(hive_test.nrow))
    assert hive_test.ncol == 3, 'Error: Number of columns are not correct.{0}'.format(str(hive_test.ncol))
    hive_test_noarg = h2o.import_file(path=pyunit_utils.locate(path_hive), destination_frame='hive_hex')
    assert hive_test_noarg.nrow == 3, 'Error: Number of rows are not correct.{0}'.format(str(hive_test_noarg.nrow))
    assert hive_test_noarg.ncol == 3, 'Error: Number of columns are not correct.{0}'.format(str(hive_test_noarg.ncol))
    path_semi = 'smalldata/parser/semi.scsv'
    semi_test = h2o.import_file(path=pyunit_utils.locate(path_semi), destination_frame='semi_hex', sep=';')
    assert semi_test.nrow == 3, 'Error: Number of rows are not correct.{0}'.format(str(semi_test.nrow))
    assert semi_test.ncol == 3, 'Error: Number of columns are not correct.{0}'.format(str(semi_test.ncol))
    semi_test_noarg = h2o.import_file(path=pyunit_utils.locate(path_semi), destination_frame='semi_hex')
    assert semi_test_noarg.nrow == 3, 'Error: Number of rows are not correct.{0}'.format(str(semi_test_noarg.nrow))
    assert semi_test_noarg.ncol == 3, 'Error: Number of columns are not correct.{0}'.format(str(semi_test_noarg.ncol))
    path_caret = 'smalldata/parser/caret.csv'
    caret_test = h2o.import_file(path=pyunit_utils.locate(path_caret), destination_frame='caret_hex', sep=u'^')
    assert caret_test.nrow == 3 and caret_test.ncol == 3
    path_asterisk = 'smalldata/parser/asterisk.asv'
    asterisk_test = h2o.import_file(path=pyunit_utils.locate(path_asterisk), destination_frame='asterisk_hex', sep=u'*')
    assert asterisk_test.nrow == 3 and asterisk_test.ncol == 3
if __name__ == '__main__':
    pyunit_utils.standalone_test(separator_test)
else:
    separator_test()
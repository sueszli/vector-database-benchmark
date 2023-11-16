from io import StringIO
import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.table import QTable
from .common import assert_almost_equal, assert_equal

def assert_equal_splitlines(arg1, arg2):
    if False:
        while True:
            i = 10
    assert_equal(arg1.splitlines(), arg2.splitlines())

def test_read_normal():
    if False:
        i = 10
        return i + 15
    'Normal SimpleRST Table'
    table = '\n# comment (with blank line above)\n======= =========\n   Col1      Col2\n======= =========\n   1.2    "hello"\n   2.4  \'s worlds\n======= =========\n'
    reader = ascii.get_reader(reader_cls=ascii.RST)
    dat = reader.read(table)
    assert_equal(dat.colnames, ['Col1', 'Col2'])
    assert_almost_equal(dat[1][0], 2.4)
    assert_equal(dat[0][1], '"hello"')
    assert_equal(dat[1][1], "'s worlds")

def test_read_normal_names():
    if False:
        for i in range(10):
            print('nop')
    'Normal SimpleRST Table with provided column names'
    table = '\n# comment (with blank line above)\n======= =========\n   Col1      Col2\n======= =========\n   1.2    "hello"\n   2.4  \'s worlds\n======= =========\n'
    reader = ascii.get_reader(reader_cls=ascii.RST, names=('name1', 'name2'))
    dat = reader.read(table)
    assert_equal(dat.colnames, ['name1', 'name2'])
    assert_almost_equal(dat[1][0], 2.4)

def test_read_normal_names_include():
    if False:
        while True:
            i = 10
    'Normal SimpleRST Table with provided column names'
    table = '\n# comment (with blank line above)\n=======  ========== ======\n   Col1     Col2      Col3\n=======  ========== ======\n   1.2     "hello"       3\n   2.4    \'s worlds      7\n=======  ========== ======\n'
    reader = ascii.get_reader(reader_cls=ascii.RST, names=('name1', 'name2', 'name3'), include_names=('name1', 'name3'))
    dat = reader.read(table)
    assert_equal(dat.colnames, ['name1', 'name3'])
    assert_almost_equal(dat[1][0], 2.4)
    assert_equal(dat[0][1], 3)

def test_read_normal_exclude():
    if False:
        return 10
    'Nice, typical SimpleRST table with col name excluded'
    table = '\n======= ==========\n  Col1     Col2\n======= ==========\n  1.2     "hello"\n  2.4    \'s worlds\n======= ==========\n'
    reader = ascii.get_reader(reader_cls=ascii.RST, exclude_names=('Col1',))
    dat = reader.read(table)
    assert_equal(dat.colnames, ['Col2'])
    assert_equal(dat[1][0], "'s worlds")

def test_read_unbounded_right_column():
    if False:
        return 10
    'The right hand column should be allowed to overflow'
    table = '\n# comment (with blank line above)\n===== ===== ====\n Col1  Col2 Col3\n===== ===== ====\n 1.2    2    Hello\n 2.4     4   Worlds\n===== ===== ====\n'
    reader = ascii.get_reader(reader_cls=ascii.RST)
    dat = reader.read(table)
    assert_equal(dat[0][2], 'Hello')
    assert_equal(dat[1][2], 'Worlds')

def test_read_unbounded_right_column_header():
    if False:
        for i in range(10):
            print('nop')
    'The right hand column should be allowed to overflow'
    table = '\n# comment (with blank line above)\n===== ===== ====\n Col1  Col2 Col3Long\n===== ===== ====\n 1.2    2    Hello\n 2.4     4   Worlds\n===== ===== ====\n'
    reader = ascii.get_reader(reader_cls=ascii.RST)
    dat = reader.read(table)
    assert_equal(dat.colnames[-1], 'Col3Long')

def test_read_right_indented_table():
    if False:
        print('Hello World!')
    'We should be able to read right indented tables correctly'
    table = '\n# comment (with blank line above)\n   ==== ==== ====\n   Col1 Col2 Col3\n   ==== ==== ====\n    3    3.4  foo\n    1    4.5  bar\n   ==== ==== ====\n'
    reader = ascii.get_reader(reader_cls=ascii.RST)
    dat = reader.read(table)
    assert_equal(dat.colnames, ['Col1', 'Col2', 'Col3'])
    assert_equal(dat[0][2], 'foo')
    assert_equal(dat[1][0], 1)

def test_trailing_spaces_in_row_definition():
    if False:
        for i in range(10):
            print('nop')
    "Trailing spaces in the row definition column shouldn't matter"
    table = '\n# comment (with blank line above)\n   ==== ==== ====    \n   Col1 Col2 Col3\n   ==== ==== ====  \n    3    3.4  foo\n    1    4.5  bar\n   ==== ==== ====  \n'
    assert len(table) == 151
    reader = ascii.get_reader(reader_cls=ascii.RST)
    dat = reader.read(table)
    assert_equal(dat.colnames, ['Col1', 'Col2', 'Col3'])
    assert_equal(dat[0][2], 'foo')
    assert_equal(dat[1][0], 1)
table = '====== =========== ============ ===========\n  Col1    Col2        Col3        Col4\n====== =========== ============ ===========\n  1.2    "hello"      1           a\n  2.4   \'s worlds          2           2\n====== =========== ============ ===========\n'
dat = ascii.read(table, format='rst')

def test_write_normal():
    if False:
        i = 10
        return i + 15
    'Write a table as a normal SimpleRST Table'
    out = StringIO()
    ascii.write(dat, out, format='rst')
    assert_equal_splitlines(out.getvalue(), '==== ========= ==== ====\nCol1      Col2 Col3 Col4\n==== ========= ==== ====\n 1.2   "hello"    1    a\n 2.4 \'s worlds    2    2\n==== ========= ==== ====\n')

def test_rst_with_header_rows():
    if False:
        for i in range(10):
            print('nop')
    'Round-trip a table with header_rows specified'
    lines = ['======= ======== ====', '   wave response ints', '     nm       ct     ', 'float64  float32 int8', '======= ======== ====', '  350.0      1.0    1', '  950.0      2.0    2', '======= ======== ====']
    tbl = QTable.read(lines, format='ascii.rst', header_rows=['name', 'unit', 'dtype'])
    assert tbl['wave'].unit == u.nm
    assert tbl['response'].unit == u.ct
    assert tbl['wave'].dtype == np.float64
    assert tbl['response'].dtype == np.float32
    assert tbl['ints'].dtype == np.int8
    out = StringIO()
    tbl.write(out, format='ascii.rst', header_rows=['name', 'unit', 'dtype'])
    assert out.getvalue().splitlines() == lines
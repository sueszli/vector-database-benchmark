import re
from io import StringIO
import pytest
from astropy.io import ascii
from astropy.io.ascii.core import masked
from astropy.io.ascii.ipac import IpacFormatError, IpacFormatErrorDBMS
from astropy.io.ascii.ui import read
from astropy.table import Column, Table
DATA = '\n|   a  |   b   |\n| char | char  |\nABBBBBBABBBBBBBA\n'

def test_ipac_default():
    if False:
        for i in range(10):
            print('nop')
    table = read(DATA, format='ipac')
    assert table['a'][0] == 'BBBBBB'
    assert table['b'][0] == 'BBBBBBB'

def test_ipac_ignore():
    if False:
        for i in range(10):
            print('nop')
    table = read(DATA, format='ipac', definition='ignore')
    assert table['a'][0] == 'BBBBBB'
    assert table['b'][0] == 'BBBBBBB'

def test_ipac_left():
    if False:
        i = 10
        return i + 15
    table = read(DATA, format='ipac', definition='left')
    assert table['a'][0] == 'BBBBBBA'
    assert table['b'][0] == 'BBBBBBBA'

def test_ipac_right():
    if False:
        print('Hello World!')
    table = read(DATA, format='ipac', definition='right')
    assert table['a'][0] == 'ABBBBBB'
    assert table['b'][0] == 'ABBBBBBB'

def test_too_long_colname_default():
    if False:
        while True:
            i = 10
    table = Table([[3]], names=['a1234567890123456789012345678901234567890'])
    out = StringIO()
    with pytest.raises(IpacFormatError):
        ascii.write(table, out, format='ipac')

def test_too_long_colname_strict():
    if False:
        return 10
    table = Table([[3]], names=['a1234567890123456'])
    out = StringIO()
    with pytest.raises(IpacFormatErrorDBMS):
        ascii.write(table, out, format='ipac', DBMS=True)

def test_too_long_colname_notstrict():
    if False:
        i = 10
        return i + 15
    table = Table([[3]], names=['a1234567890123456789012345678901234567890'])
    out = StringIO()
    with pytest.raises(IpacFormatError):
        ascii.write(table, out, format='ipac', DBMS=False)

@pytest.mark.parametrize(('strict_', 'Err'), [(True, IpacFormatErrorDBMS), (False, IpacFormatError)])
def test_non_alfnum_colname(strict_, Err):
    if False:
        return 10
    table = Table([[3]], names=['a123456789 01234'])
    out = StringIO()
    with pytest.raises(Err):
        ascii.write(table, out, format='ipac', DBMS=strict_)

def test_colname_starswithnumber_strict():
    if False:
        return 10
    table = Table([[3]], names=['a123456789 01234'])
    out = StringIO()
    with pytest.raises(IpacFormatErrorDBMS):
        ascii.write(table, out, format='ipac', DBMS=True)

def test_double_colname_strict():
    if False:
        print('Hello World!')
    table = Table([[3], [1]], names=['DEC', 'dec'])
    out = StringIO()
    with pytest.raises(IpacFormatErrorDBMS):
        ascii.write(table, out, format='ipac', DBMS=True)

@pytest.mark.parametrize('colname', ['x', 'y', 'z', 'X', 'Y', 'Z'])
def test_reserved_colname_strict(colname):
    if False:
        print('Hello World!')
    table = Table([['reg']], names=[colname])
    out = StringIO()
    with pytest.raises(IpacFormatErrorDBMS):
        ascii.write(table, out, format='ipac', DBMS=True)

def test_too_long_comment():
    if False:
        while True:
            i = 10
    msg = 'Wrapping comment lines > 78 characters produced 1 extra line(s)'
    with pytest.warns(UserWarning, match=re.escape(msg)):
        table = Table([[3]])
        table.meta['comments'] = ['a' * 79]
        out = StringIO()
        ascii.write(table, out, format='ipac')
    expected_out = '\\ aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\\ a\n|col0|\n|long|\n|    |\n|null|\n    3\n'
    assert out.getvalue().strip().splitlines() == expected_out.splitlines()

def test_out_with_nonstring_null():
    if False:
        print('Hello World!')
    'Test a (non-string) fill value.\n\n    Even for an unmasked tables, the fill_value should show up in the\n    table header.\n    '
    table = Table([[3]], masked=True)
    out = StringIO()
    ascii.write(table, out, format='ipac', fill_values=[(masked, -99999)])
    expected_out = '|  col0|\n|  long|\n|      |\n|-99999|\n      3\n'
    assert out.getvalue().strip().splitlines() == expected_out.splitlines()

def test_include_exclude_names():
    if False:
        while True:
            i = 10
    table = Table([[1], [2], [3]], names=('A', 'B', 'C'))
    out = StringIO()
    ascii.write(table, out, format='ipac', include_names=('A', 'B'), exclude_names=('A',))
    expected_out = '|   B|\n|long|\n|    |\n|null|\n    2\n'
    assert out.getvalue().strip().splitlines() == expected_out.splitlines()

def test_short_dtypes():
    if False:
        print('Hello World!')
    table = Table([Column([1.0], dtype='f4'), Column([2], dtype='i2')], names=('float_col', 'int_col'))
    out = StringIO()
    ascii.write(table, out, format='ipac')
    expected_out = '|float_col|int_col|\n|    float|    int|\n|         |       |\n|     null|   null|\n       1.0       2\n'
    assert out.getvalue().strip().splitlines() == expected_out.splitlines()
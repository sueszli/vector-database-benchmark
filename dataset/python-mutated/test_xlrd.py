import io
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.io.excel import ExcelFile
from pandas.io.excel._base import inspect_excel_format
xlrd = pytest.importorskip('xlrd')

@pytest.fixture(params=['.xls'])
def read_ext_xlrd(request):
    if False:
        print('Hello World!')
    '\n    Valid extensions for reading Excel files with xlrd.\n\n    Similar to read_ext, but excludes .ods, .xlsb, and for xlrd>2 .xlsx, .xlsm\n    '
    return request.param

def test_read_xlrd_book(read_ext_xlrd, datapath):
    if False:
        print('Hello World!')
    engine = 'xlrd'
    sheet_name = 'Sheet1'
    pth = datapath('io', 'data', 'excel', 'test1.xls')
    with xlrd.open_workbook(pth) as book:
        with ExcelFile(book, engine=engine) as xl:
            result = pd.read_excel(xl, sheet_name=sheet_name, index_col=0)
        expected = pd.read_excel(book, sheet_name=sheet_name, engine=engine, index_col=0)
    tm.assert_frame_equal(result, expected)

def test_read_xlsx_fails(datapath):
    if False:
        while True:
            i = 10
    from xlrd.biffh import XLRDError
    path = datapath('io', 'data', 'excel', 'test1.xlsx')
    with pytest.raises(XLRDError, match='Excel xlsx file; not supported'):
        pd.read_excel(path, engine='xlrd')

def test_nan_in_xls(datapath):
    if False:
        return 10
    path = datapath('io', 'data', 'excel', 'test6.xls')
    expected = pd.DataFrame({0: np.r_[0, 2].astype('int64'), 1: np.r_[1, np.nan]})
    result = pd.read_excel(path, header=None)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('file_header', [b'\t\x00\x04\x00\x07\x00\x10\x00', b'\t\x02\x06\x00\x00\x00\x10\x00', b'\t\x04\x06\x00\x00\x00\x10\x00', b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'])
def test_read_old_xls_files(file_header):
    if False:
        while True:
            i = 10
    f = io.BytesIO(file_header)
    assert inspect_excel_format(f) == 'xls'
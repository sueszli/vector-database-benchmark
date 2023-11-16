"""
Tests that apply specifically to the CParser. Unless specifically stated
as a CParser-specific issue, the goal is to eventually move as many of
these tests out of this module as soon as the Python parser can accept
further arguments when parsing.
"""
from decimal import Decimal
from io import BytesIO, StringIO, TextIOWrapper
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat import is_ci_environment
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import ParserError, ParserWarning
import pandas.util._test_decorators as td
from pandas import DataFrame, concat
import pandas._testing as tm

@pytest.mark.parametrize('malformed', ['1\r1\r1\r 1\r 1\r', '1\r1\r1\r 1\r 1\r11\r', '1\r1\r1\r 1\r 1\r11\r1\r'], ids=['words pointer', 'stream pointer', 'lines pointer'])
def test_buffer_overflow(c_parser_only, malformed):
    if False:
        while True:
            i = 10
    msg = 'Buffer overflow caught - possible malformed input file.'
    parser = c_parser_only
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(malformed))

def test_delim_whitespace_custom_terminator(c_parser_only):
    if False:
        while True:
            i = 10
    data = 'a b c~1 2 3~4 5 6~7 8 9'
    parser = c_parser_only
    df = parser.read_csv(StringIO(data), lineterminator='~', delim_whitespace=True)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['a', 'b', 'c'])
    tm.assert_frame_equal(df, expected)

def test_dtype_and_names_error(c_parser_only):
    if False:
        print('Hello World!')
    parser = c_parser_only
    data = '\n1.0 1\n2.0 2\n3.0 3\n'
    result = parser.read_csv(StringIO(data), sep='\\s+', header=None)
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]])
    tm.assert_frame_equal(result, expected)
    result = parser.read_csv(StringIO(data), sep='\\s+', header=None, names=['a', 'b'])
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)
    result = parser.read_csv(StringIO(data), sep='\\s+', header=None, names=['a', 'b'], dtype={'a': np.int32})
    expected = DataFrame([[1, 1], [2, 2], [3, 3]], columns=['a', 'b'])
    expected['a'] = expected['a'].astype(np.int32)
    tm.assert_frame_equal(result, expected)
    data = '\n1.0 1\nnan 2\n3.0 3\n'
    warning = RuntimeWarning if np_version_gte1p24 else None
    with pytest.raises(ValueError, match='cannot safely convert'):
        with tm.assert_produces_warning(warning, check_stacklevel=False):
            parser.read_csv(StringIO(data), sep='\\s+', header=None, names=['a', 'b'], dtype={'a': np.int32})

@pytest.mark.parametrize('match,kwargs', [('the dtype datetime64 is not supported for parsing, pass this column using parse_dates instead', {'dtype': {'A': 'datetime64', 'B': 'float64'}}), ('the dtype datetime64 is not supported for parsing, pass this column using parse_dates instead', {'dtype': {'A': 'datetime64', 'B': 'float64'}, 'parse_dates': ['B']}), ('the dtype timedelta64 is not supported for parsing', {'dtype': {'A': 'timedelta64', 'B': 'float64'}}), (f'the dtype {tm.ENDIAN}U8 is not supported for parsing', {'dtype': {'A': 'U8'}})], ids=['dt64-0', 'dt64-1', 'td64', f'{tm.ENDIAN}U8'])
def test_unsupported_dtype(c_parser_only, match, kwargs):
    if False:
        print('Hello World!')
    parser = c_parser_only
    df = DataFrame(np.random.default_rng(2).random((5, 2)), columns=list('AB'), index=['1A', '1B', '1C', '1D', '1E'])
    with tm.ensure_clean('__unsupported_dtype__.csv') as path:
        df.to_csv(path)
        with pytest.raises(TypeError, match=match):
            parser.read_csv(path, index_col=0, **kwargs)

@td.skip_if_32bit
@pytest.mark.slow
@pytest.mark.parametrize('num', np.linspace(1.0, 2.0, num=21))
def test_precise_conversion(c_parser_only, num):
    if False:
        print('Hello World!')
    parser = c_parser_only
    normal_errors = []
    precise_errors = []

    def error(val: float, actual_val: Decimal) -> Decimal:
        if False:
            while True:
                i = 10
        return abs(Decimal(f'{val:.100}') - actual_val)
    text = f'a\n{num:.25}'
    normal_val = float(parser.read_csv(StringIO(text), float_precision='legacy')['a'][0])
    precise_val = float(parser.read_csv(StringIO(text), float_precision='high')['a'][0])
    roundtrip_val = float(parser.read_csv(StringIO(text), float_precision='round_trip')['a'][0])
    actual_val = Decimal(text[2:])
    normal_errors.append(error(normal_val, actual_val))
    precise_errors.append(error(precise_val, actual_val))
    assert roundtrip_val == float(text[2:])
    assert sum(precise_errors) <= sum(normal_errors)
    assert max(precise_errors) <= max(normal_errors)

def test_usecols_dtypes(c_parser_only):
    if False:
        for i in range(10):
            print('nop')
    parser = c_parser_only
    data = '1,2,3\n4,5,6\n7,8,9\n10,11,12'
    result = parser.read_csv(StringIO(data), usecols=(0, 1, 2), names=('a', 'b', 'c'), header=None, converters={'a': str}, dtype={'b': int, 'c': float})
    result2 = parser.read_csv(StringIO(data), usecols=(0, 2), names=('a', 'b', 'c'), header=None, converters={'a': str}, dtype={'b': int, 'c': float})
    assert (result.dtypes == [object, int, float]).all()
    assert (result2.dtypes == [object, float]).all()

def test_disable_bool_parsing(c_parser_only):
    if False:
        for i in range(10):
            print('nop')
    parser = c_parser_only
    data = 'A,B,C\nYes,No,Yes\nNo,Yes,Yes\nYes,,Yes\nNo,No,No'
    result = parser.read_csv(StringIO(data), dtype=object)
    assert (result.dtypes == object).all()
    result = parser.read_csv(StringIO(data), dtype=object, na_filter=False)
    assert result['B'][2] == ''

def test_custom_lineterminator(c_parser_only):
    if False:
        i = 10
        return i + 15
    parser = c_parser_only
    data = 'a,b,c~1,2,3~4,5,6'
    result = parser.read_csv(StringIO(data), lineterminator='~')
    expected = parser.read_csv(StringIO(data.replace('~', '\n')))
    tm.assert_frame_equal(result, expected)

def test_parse_ragged_csv(c_parser_only):
    if False:
        while True:
            i = 10
    parser = c_parser_only
    data = '1,2,3\n1,2,3,4\n1,2,3,4,5\n1,2\n1,2,3,4'
    nice_data = '1,2,3,,\n1,2,3,4,\n1,2,3,4,5\n1,2,,,\n1,2,3,4,'
    result = parser.read_csv(StringIO(data), header=None, names=['a', 'b', 'c', 'd', 'e'])
    expected = parser.read_csv(StringIO(nice_data), header=None, names=['a', 'b', 'c', 'd', 'e'])
    tm.assert_frame_equal(result, expected)
    data = '1,2\n3,4,5'
    result = parser.read_csv(StringIO(data), header=None, names=range(50))
    expected = parser.read_csv(StringIO(data), header=None, names=range(3)).reindex(columns=range(50))
    tm.assert_frame_equal(result, expected)

def test_tokenize_CR_with_quoting(c_parser_only):
    if False:
        print('Hello World!')
    parser = c_parser_only
    data = ' a,b,c\r"a,b","e,d","f,f"'
    result = parser.read_csv(StringIO(data), header=None)
    expected = parser.read_csv(StringIO(data.replace('\r', '\n')), header=None)
    tm.assert_frame_equal(result, expected)
    result = parser.read_csv(StringIO(data))
    expected = parser.read_csv(StringIO(data.replace('\r', '\n')))
    tm.assert_frame_equal(result, expected)

@pytest.mark.slow
@pytest.mark.parametrize('count', [3 * 2 ** n for n in range(6)])
def test_grow_boundary_at_cap(c_parser_only, count):
    if False:
        print('Hello World!')
    parser = c_parser_only
    with StringIO(',' * count) as s:
        expected = DataFrame(columns=[f'Unnamed: {i}' for i in range(count + 1)])
        df = parser.read_csv(s)
    tm.assert_frame_equal(df, expected)

@pytest.mark.slow
@pytest.mark.parametrize('encoding', [None, 'utf-8'])
def test_parse_trim_buffers(c_parser_only, encoding):
    if False:
        print('Hello World!')
    parser = c_parser_only
    record_ = '9999-9,99:99,,,,ZZ,ZZ,,,ZZZ-ZZZZ,.Z-ZZZZ,-9.99,,,9.99,ZZZZZ,,-99,9,ZZZ-ZZZZ,ZZ-ZZZZ,,9.99,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,999,ZZZ-ZZZZ,,ZZ-ZZZZ,,,,,ZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,,,9,9,9,9,99,99,999,999,ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,9,ZZ-ZZZZ,9.99,ZZ-ZZZZ,ZZ-ZZZZ,,,,ZZZZ,,,ZZ,ZZ,,,,,,,,,,,,,9,,,999.99,999.99,,,ZZZZZ,,,Z9,,,,,,,ZZZ,ZZZ,,,,,,,,,,,ZZZZZ,ZZZZZ,ZZZ-ZZZZZZ,ZZZ-ZZZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,,,999999,999999,ZZZ,ZZZ,,,ZZZ,ZZZ,999.99,999.99,,,,ZZZ-ZZZ,ZZZ-ZZZ,-9.99,-9.99,9,9,,99,,9.99,9.99,9,9,9.99,9.99,,,,9.99,9.99,,99,,99,9.99,9.99,,,ZZZ,ZZZ,,999.99,,999.99,ZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,,,ZZZZZ,ZZZZZ,ZZZ,ZZZ,9,9,,,,,,ZZZ-ZZZZ,ZZZ999Z,,,999.99,,999.99,ZZZ-ZZZZ,,,9.999,9.999,9.999,9.999,-9.999,-9.999,-9.999,-9.999,9.999,9.999,9.999,9.999,9.999,9.999,9.999,9.999,99999,ZZZ-ZZZZ,,9.99,ZZZ,,,,,,,,ZZZ,,,,,9,,,,9,,,,,,,,,,ZZZ-ZZZZ,ZZZ-ZZZZ,,ZZZZZ,ZZZZZ,ZZZZZ,ZZZZZ,,,9.99,,ZZ-ZZZZ,ZZ-ZZZZ,ZZ,999,,,,ZZ-ZZZZ,ZZZ,ZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,,,99.99,99.99,,,9.99,9.99,9.99,9.99,ZZZ-ZZZZ,,,ZZZ-ZZZZZ,,,,,-9.99,-9.99,-9.99,-9.99,,,,,,,,,ZZZ-ZZZZ,,9,9.99,9.99,99ZZ,,-9.99,-9.99,ZZZ-ZZZZ,,,,,,,ZZZ-ZZZZ,9.99,9.99,9999,,,,,,,,,,-9.9,Z/Z-ZZZZ,999.99,9.99,,999.99,ZZ-ZZZZ,ZZ-ZZZZ,9.99,9.99,9.99,9.99,9.99,9.99,,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ,ZZZ,ZZZ,ZZZ,9.99,,,-9.99,ZZ-ZZZZ,-999.99,,-9999,,999.99,,,,999.99,99.99,,,ZZ-ZZZZZZZZ,ZZ-ZZZZ-ZZZZZZZ,,,,ZZ-ZZ-ZZZZZZZZ,ZZZZZZZZ,ZZZ-ZZZZ,9999,999.99,ZZZ-ZZZZ,-9.99,-9.99,ZZZ-ZZZZ,99:99:99,,99,99,,9.99,,-99.99,,,,,,9.99,ZZZ-ZZZZ,-9.99,-9.99,9.99,9.99,,ZZZ,,,,,,,ZZZ,ZZZ,,,,,'
    (chunksize, n_lines) = (128, 2 * 128 + 15)
    csv_data = '\n'.join([record_] * n_lines) + '\n'
    row = tuple((val_ if val_ else np.nan for val_ in record_.split(',')))
    expected = DataFrame([row for _ in range(n_lines)], dtype=object, columns=None, index=None)
    with parser.read_csv(StringIO(csv_data), header=None, dtype=object, chunksize=chunksize, encoding=encoding) as chunks_:
        result = concat(chunks_, axis=0, ignore_index=True)
    tm.assert_frame_equal(result, expected)

def test_internal_null_byte(c_parser_only):
    if False:
        return 10
    parser = c_parser_only
    names = ['a', 'b', 'c']
    data = '1,2,3\n4,\x00,6\n7,8,9'
    expected = DataFrame([[1, 2.0, 3], [4, np.nan, 6], [7, 8, 9]], columns=names)
    result = parser.read_csv(StringIO(data), names=names)
    tm.assert_frame_equal(result, expected)

def test_read_nrows_large(c_parser_only):
    if False:
        return 10
    parser = c_parser_only
    header_narrow = '\t'.join(['COL_HEADER_' + str(i) for i in range(10)]) + '\n'
    data_narrow = '\t'.join(['somedatasomedatasomedata1' for _ in range(10)]) + '\n'
    header_wide = '\t'.join(['COL_HEADER_' + str(i) for i in range(15)]) + '\n'
    data_wide = '\t'.join(['somedatasomedatasomedata2' for _ in range(15)]) + '\n'
    test_input = header_narrow + data_narrow * 1050 + header_wide + data_wide * 2
    df = parser.read_csv(StringIO(test_input), sep='\t', nrows=1010)
    assert df.size == 1010 * 10

def test_float_precision_round_trip_with_text(c_parser_only):
    if False:
        return 10
    parser = c_parser_only
    df = parser.read_csv(StringIO('a'), header=None, float_precision='round_trip')
    tm.assert_frame_equal(df, DataFrame({0: ['a']}))

def test_large_difference_in_columns(c_parser_only):
    if False:
        return 10
    parser = c_parser_only
    count = 10000
    large_row = ('X,' * count)[:-1] + '\n'
    normal_row = 'XXXXXX XXXXXX,111111111111111\n'
    test_input = (large_row + normal_row * 6)[:-1]
    result = parser.read_csv(StringIO(test_input), header=None, usecols=[0])
    rows = test_input.split('\n')
    expected = DataFrame([row.split(',')[0] for row in rows])
    tm.assert_frame_equal(result, expected)

def test_data_after_quote(c_parser_only):
    if False:
        while True:
            i = 10
    parser = c_parser_only
    data = 'a\n1\n"b"a'
    result = parser.read_csv(StringIO(data))
    expected = DataFrame({'a': ['1', 'ba']})
    tm.assert_frame_equal(result, expected)

def test_comment_whitespace_delimited(c_parser_only):
    if False:
        print('Hello World!')
    parser = c_parser_only
    test_input = '1 2\n2 2 3\n3 2 3 # 3 fields\n4 2 3# 3 fields\n5 2 # 2 fields\n6 2# 2 fields\n7 # 1 field, NaN\n8# 1 field, NaN\n9 2 3 # skipped line\n# comment'
    with tm.assert_produces_warning(ParserWarning, match='Skipping line', check_stacklevel=False):
        df = parser.read_csv(StringIO(test_input), comment='#', header=None, delimiter='\\s+', skiprows=0, on_bad_lines='warn')
    expected = DataFrame([[1, 2], [5, 2], [6, 2], [7, np.nan], [8, np.nan]])
    tm.assert_frame_equal(df, expected)

def test_file_like_no_next(c_parser_only):
    if False:
        while True:
            i = 10

    class NoNextBuffer(StringIO):

        def __next__(self):
            if False:
                i = 10
                return i + 15
            raise AttributeError('No next method')
        next = __next__
    parser = c_parser_only
    data = 'a\n1'
    expected = DataFrame({'a': [1]})
    result = parser.read_csv(NoNextBuffer(data))
    tm.assert_frame_equal(result, expected)

def test_buffer_rd_bytes_bad_unicode(c_parser_only):
    if False:
        for i in range(10):
            print('nop')
    t = BytesIO(b'\xb0')
    t = TextIOWrapper(t, encoding='ascii', errors='surrogateescape')
    msg = "'utf-8' codec can't encode character"
    with pytest.raises(UnicodeError, match=msg):
        c_parser_only.read_csv(t, encoding='UTF-8')

@pytest.mark.parametrize('tar_suffix', ['.tar', '.tar.gz'])
def test_read_tarfile(c_parser_only, csv_dir_path, tar_suffix):
    if False:
        i = 10
        return i + 15
    parser = c_parser_only
    tar_path = os.path.join(csv_dir_path, 'tar_csv' + tar_suffix)
    with tarfile.open(tar_path, 'r') as tar:
        data_file = tar.extractfile('tar_data.csv')
        out = parser.read_csv(data_file)
        expected = DataFrame({'a': [1]})
        tm.assert_frame_equal(out, expected)

@pytest.mark.single_cpu
@pytest.mark.skipif(is_ci_environment(), reason='Too memory intensive for CI.')
def test_bytes_exceed_2gb(c_parser_only):
    if False:
        i = 10
        return i + 15
    parser = c_parser_only
    if parser.low_memory:
        pytest.skip('not a low_memory test')
    csv = StringIO('strings\n' + '\n'.join(['x' * (1 << 20) for _ in range(2100)]))
    df = parser.read_csv(csv)
    assert not df.empty

def test_chunk_whitespace_on_boundary(c_parser_only):
    if False:
        for i in range(10):
            print('nop')
    parser = c_parser_only
    chunk1 = 'a' * (1024 * 256 - 2) + '\na'
    chunk2 = '\n a'
    result = parser.read_csv(StringIO(chunk1 + chunk2), header=None)
    expected = DataFrame(['a' * (1024 * 256 - 2), 'a', ' a'])
    tm.assert_frame_equal(result, expected)

def test_file_handles_mmap(c_parser_only, csv1):
    if False:
        for i in range(10):
            print('nop')
    parser = c_parser_only
    with open(csv1, encoding='utf-8') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            parser.read_csv(m)
            assert not m.closed

def test_file_binary_mode(c_parser_only):
    if False:
        for i in range(10):
            print('nop')
    parser = c_parser_only
    expected = DataFrame([[1, 2, 3], [4, 5, 6]])
    with tm.ensure_clean() as path:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('1,2,3\n4,5,6')
        with open(path, 'rb') as f:
            result = parser.read_csv(f, header=None)
            tm.assert_frame_equal(result, expected)

def test_unix_style_breaks(c_parser_only):
    if False:
        return 10
    parser = c_parser_only
    with tm.ensure_clean() as path:
        with open(path, 'w', newline='\n', encoding='utf-8') as f:
            f.write('blah\n\ncol_1,col_2,col_3\n\n')
        result = parser.read_csv(path, skiprows=2, encoding='utf-8', engine='c')
    expected = DataFrame(columns=['col_1', 'col_2', 'col_3'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('float_precision', [None, 'legacy', 'high', 'round_trip'])
@pytest.mark.parametrize('data,thousands,decimal', [('A|B|C\n1|2,334.01|5\n10|13|10.\n', ',', '.'), ('A|B|C\n1|2.334,01|5\n10|13|10,\n', '.', ',')])
def test_1000_sep_with_decimal(c_parser_only, data, thousands, decimal, float_precision):
    if False:
        return 10
    parser = c_parser_only
    expected = DataFrame({'A': [1, 10], 'B': [2334.01, 13], 'C': [5, 10.0]})
    result = parser.read_csv(StringIO(data), sep='|', thousands=thousands, decimal=decimal, float_precision=float_precision)
    tm.assert_frame_equal(result, expected)

def test_float_precision_options(c_parser_only):
    if False:
        return 10
    parser = c_parser_only
    s = 'foo\n243.164\n'
    df = parser.read_csv(StringIO(s))
    df2 = parser.read_csv(StringIO(s), float_precision='high')
    tm.assert_frame_equal(df, df2)
    df3 = parser.read_csv(StringIO(s), float_precision='legacy')
    assert not df.iloc[0, 0] == df3.iloc[0, 0]
    msg = 'Unrecognized float_precision option: junk'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(s), float_precision='junk')
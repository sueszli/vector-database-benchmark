"""
Tests that apply specifically to the Python parser. Unless specifically
stated as a Python-specific issue, the goal is to eventually move as many of
these tests out of this module as soon as the C parser can accept further
arguments when parsing.
"""
from __future__ import annotations
import csv
from io import BytesIO, StringIO, TextIOWrapper
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import ParserError, ParserWarning
from pandas import DataFrame, Index, MultiIndex
import pandas._testing as tm
if TYPE_CHECKING:
    from collections.abc import Iterator

def test_default_separator(python_parser_only):
    if False:
        while True:
            i = 10
    data = 'aob\n1o2\n3o4'
    parser = python_parser_only
    expected = DataFrame({'a': [1, 3], 'b': [2, 4]})
    result = parser.read_csv(StringIO(data), sep=None)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('skipfooter', ['foo', 1.5, True])
def test_invalid_skipfooter_non_int(python_parser_only, skipfooter):
    if False:
        for i in range(10):
            print('nop')
    data = 'a\n1\n2'
    parser = python_parser_only
    msg = 'skipfooter must be an integer'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=skipfooter)

def test_invalid_skipfooter_negative(python_parser_only):
    if False:
        print('Hello World!')
    data = 'a\n1\n2'
    parser = python_parser_only
    msg = 'skipfooter cannot be negative'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=-1)

@pytest.mark.parametrize('kwargs', [{'sep': None}, {'delimiter': '|'}])
def test_sniff_delimiter(python_parser_only, kwargs):
    if False:
        print('Hello World!')
    data = 'index|A|B|C\nfoo|1|2|3\nbar|4|5|6\nbaz|7|8|9\n'
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'], index=Index(['foo', 'bar', 'baz'], name='index'))
    tm.assert_frame_equal(result, expected)

def test_sniff_delimiter_comment(python_parser_only):
    if False:
        i = 10
        return i + 15
    data = '# comment line\nindex|A|B|C\n# comment line\nfoo|1|2|3 # ignore | this\nbar|4|5|6\nbaz|7|8|9\n'
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), index_col=0, sep=None, comment='#')
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'], index=Index(['foo', 'bar', 'baz'], name='index'))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('encoding', [None, 'utf-8'])
def test_sniff_delimiter_encoding(python_parser_only, encoding):
    if False:
        print('Hello World!')
    parser = python_parser_only
    data = 'ignore this\nignore this too\nindex|A|B|C\nfoo|1|2|3\nbar|4|5|6\nbaz|7|8|9\n'
    if encoding is not None:
        data = data.encode(encoding)
        data = BytesIO(data)
        data = TextIOWrapper(data, encoding=encoding)
    else:
        data = StringIO(data)
    result = parser.read_csv(data, index_col=0, sep=None, skiprows=2, encoding=encoding)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'], index=Index(['foo', 'bar', 'baz'], name='index'))
    tm.assert_frame_equal(result, expected)

def test_single_line(python_parser_only):
    if False:
        while True:
            i = 10
    parser = python_parser_only
    result = parser.read_csv(StringIO('1,2'), names=['a', 'b'], header=None, sep=None)
    expected = DataFrame({'a': [1], 'b': [2]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('kwargs', [{'skipfooter': 2}, {'nrows': 3}])
def test_skipfooter(python_parser_only, kwargs):
    if False:
        for i in range(10):
            print('nop')
    data = 'A,B,C\n1,2,3\n4,5,6\n7,8,9\nwant to skip this\nalso also skip this\n'
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), **kwargs)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('compression,klass', [('gzip', 'GzipFile'), ('bz2', 'BZ2File')])
def test_decompression_regex_sep(python_parser_only, csv1, compression, klass):
    if False:
        print('Hello World!')
    parser = python_parser_only
    with open(csv1, 'rb') as f:
        data = f.read()
    data = data.replace(b',', b'::')
    expected = parser.read_csv(csv1)
    module = pytest.importorskip(compression)
    klass = getattr(module, klass)
    with tm.ensure_clean() as path:
        with klass(path, mode='wb') as tmp:
            tmp.write(data)
        result = parser.read_csv(path, sep='::', compression=compression)
        tm.assert_frame_equal(result, expected)

def test_read_csv_buglet_4x_multi_index(python_parser_only):
    if False:
        print('Hello World!')
    data = '                      A       B       C       D        E\none two three   four\na   b   10.0032 5    -0.5109 -2.3358 -0.4645  0.05076  0.3640\na   q   20      4     0.4473  1.4152  0.2834  1.00661  0.1744\nx   q   30      3    -0.6662 -0.5243 -0.3580  0.89145  2.5838'
    parser = python_parser_only
    expected = DataFrame([[-0.5109, -2.3358, -0.4645, 0.05076, 0.364], [0.4473, 1.4152, 0.2834, 1.00661, 0.1744], [-0.6662, -0.5243, -0.358, 0.89145, 2.5838]], columns=['A', 'B', 'C', 'D', 'E'], index=MultiIndex.from_tuples([('a', 'b', 10.0032, 5), ('a', 'q', 20, 4), ('x', 'q', 30, 3)], names=['one', 'two', 'three', 'four']))
    result = parser.read_csv(StringIO(data), sep='\\s+')
    tm.assert_frame_equal(result, expected)

def test_read_csv_buglet_4x_multi_index2(python_parser_only):
    if False:
        print('Hello World!')
    data = '      A B C\na b c\n1 3 7 0 3 6\n3 1 4 1 5 9'
    parser = python_parser_only
    expected = DataFrame.from_records([(1, 3, 7, 0, 3, 6), (3, 1, 4, 1, 5, 9)], columns=list('abcABC'), index=list('abc'))
    result = parser.read_csv(StringIO(data), sep='\\s+')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('add_footer', [True, False])
def test_skipfooter_with_decimal(python_parser_only, add_footer):
    if False:
        for i in range(10):
            print('nop')
    data = '1#2\n3#4'
    parser = python_parser_only
    expected = DataFrame({'a': [1.2, 3.4]})
    if add_footer:
        kwargs = {'skipfooter': 1}
        data += '\nFooter'
    else:
        kwargs = {}
    result = parser.read_csv(StringIO(data), names=['a'], decimal='#', **kwargs)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('sep', ['::', '#####', '!!!', '123', '#1!c5', '%!c!d', '@@#4:2', '_!pd#_'])
@pytest.mark.parametrize('encoding', ['utf-16', 'utf-16-be', 'utf-16-le', 'utf-32', 'cp037'])
def test_encoding_non_utf8_multichar_sep(python_parser_only, sep, encoding):
    if False:
        return 10
    expected = DataFrame({'a': [1], 'b': [2]})
    parser = python_parser_only
    data = '1' + sep + '2'
    encoded_data = data.encode(encoding)
    result = parser.read_csv(BytesIO(encoded_data), sep=sep, names=['a', 'b'], encoding=encoding)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('quoting', [csv.QUOTE_MINIMAL, csv.QUOTE_NONE])
def test_multi_char_sep_quotes(python_parser_only, quoting):
    if False:
        return 10
    kwargs = {'sep': ',,'}
    parser = python_parser_only
    data = 'a,,b\n1,,a\n2,,"2,,b"'
    if quoting == csv.QUOTE_NONE:
        msg = 'Expected 2 fields in line 3, saw 3'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)
    else:
        msg = 'ignored when a multi-char delimiter is used'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)

def test_none_delimiter(python_parser_only):
    if False:
        print('Hello World!')
    parser = python_parser_only
    data = 'a,b,c\n0,1,2\n3,4,5,6\n7,8,9'
    expected = DataFrame({'a': [0, 7], 'b': [1, 8], 'c': [2, 9]})
    with tm.assert_produces_warning(ParserWarning, match='Skipping line 3', check_stacklevel=False):
        result = parser.read_csv(StringIO(data), header=0, sep=None, on_bad_lines='warn')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data', ['a\n1\n"b"a', 'a,b,c\ncat,foo,bar\ndog,foo,"baz'])
@pytest.mark.parametrize('skipfooter', [0, 1])
def test_skipfooter_bad_row(python_parser_only, data, skipfooter):
    if False:
        i = 10
        return i + 15
    parser = python_parser_only
    if skipfooter:
        msg = 'parsing errors in the skipped footer rows'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)
    else:
        msg = 'unexpected end of data|expected after'
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)

def test_malformed_skipfooter(python_parser_only):
    if False:
        for i in range(10):
            print('nop')
    parser = python_parser_only
    data = 'ignore\nA,B,C\n1,2,3 # comment\n1,2,3,4,5\n2,3,4\nfooter\n'
    msg = 'Expected 3 fields in line 4, saw 5'
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), header=1, comment='#', skipfooter=1)

def test_python_engine_file_no_next(python_parser_only):
    if False:
        for i in range(10):
            print('nop')
    parser = python_parser_only

    class NoNextBuffer:

        def __init__(self, csv_data) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.data = csv_data

        def __iter__(self) -> Iterator:
            if False:
                return 10
            return self.data.__iter__()

        def read(self):
            if False:
                return 10
            return self.data

        def readline(self):
            if False:
                return 10
            return self.data
    parser.read_csv(NoNextBuffer('a\n1'))

@pytest.mark.parametrize('bad_line_func', [lambda x: ['2', '3'], lambda x: x[:2]])
def test_on_bad_lines_callable(python_parser_only, bad_line_func):
    if False:
        print('Hello World!')
    parser = python_parser_only
    data = 'a,b\n1,2\n2,3,4,5,6\n3,4\n'
    bad_sio = StringIO(data)
    result = parser.read_csv(bad_sio, on_bad_lines=bad_line_func)
    expected = DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
    tm.assert_frame_equal(result, expected)

def test_on_bad_lines_callable_write_to_external_list(python_parser_only):
    if False:
        print('Hello World!')
    parser = python_parser_only
    data = 'a,b\n1,2\n2,3,4,5,6\n3,4\n'
    bad_sio = StringIO(data)
    lst = []

    def bad_line_func(bad_line: list[str]) -> list[str]:
        if False:
            return 10
        lst.append(bad_line)
        return ['2', '3']
    result = parser.read_csv(bad_sio, on_bad_lines=bad_line_func)
    expected = DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
    tm.assert_frame_equal(result, expected)
    assert lst == [['2', '3', '4', '5', '6']]

@pytest.mark.parametrize('bad_line_func', [lambda x: ['foo', 'bar'], lambda x: x[:2]])
@pytest.mark.parametrize('sep', [',', '111'])
def test_on_bad_lines_callable_iterator_true(python_parser_only, bad_line_func, sep):
    if False:
        return 10
    parser = python_parser_only
    data = f'\n0{sep}1\nhi{sep}there\nfoo{sep}bar{sep}baz\ngood{sep}bye\n'
    bad_sio = StringIO(data)
    result_iter = parser.read_csv(bad_sio, on_bad_lines=bad_line_func, chunksize=1, iterator=True, sep=sep)
    expecteds = [{'0': 'hi', '1': 'there'}, {'0': 'foo', '1': 'bar'}, {'0': 'good', '1': 'bye'}]
    for (i, (result, expected)) in enumerate(zip(result_iter, expecteds)):
        expected = DataFrame(expected, index=range(i, i + 1))
        tm.assert_frame_equal(result, expected)

def test_on_bad_lines_callable_dont_swallow_errors(python_parser_only):
    if False:
        print('Hello World!')
    parser = python_parser_only
    data = 'a,b\n1,2\n2,3,4,5,6\n3,4\n'
    bad_sio = StringIO(data)
    msg = 'This function is buggy.'

    def bad_line_func(bad_line):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError(msg)
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(bad_sio, on_bad_lines=bad_line_func)

def test_on_bad_lines_callable_not_expected_length(python_parser_only):
    if False:
        for i in range(10):
            print('nop')
    parser = python_parser_only
    data = 'a,b\n1,2\n2,3,4,5,6\n3,4\n'
    bad_sio = StringIO(data)
    result = parser.read_csv_check_warnings(ParserWarning, 'Length of header or names', bad_sio, on_bad_lines=lambda x: x)
    expected = DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
    tm.assert_frame_equal(result, expected)

def test_on_bad_lines_callable_returns_none(python_parser_only):
    if False:
        return 10
    parser = python_parser_only
    data = 'a,b\n1,2\n2,3,4,5,6\n3,4\n'
    bad_sio = StringIO(data)
    result = parser.read_csv(bad_sio, on_bad_lines=lambda x: None)
    expected = DataFrame({'a': [1, 3], 'b': [2, 4]})
    tm.assert_frame_equal(result, expected)

def test_on_bad_lines_index_col_inferred(python_parser_only):
    if False:
        while True:
            i = 10
    parser = python_parser_only
    data = 'a,b\n1,2,3\n4,5,6\n'
    bad_sio = StringIO(data)
    result = parser.read_csv(bad_sio, on_bad_lines=lambda x: ['99', '99'])
    expected = DataFrame({'a': [2, 5], 'b': [3, 6]}, index=[1, 4])
    tm.assert_frame_equal(result, expected)

def test_index_col_false_and_header_none(python_parser_only):
    if False:
        return 10
    parser = python_parser_only
    data = '\n0.5,0.03\n0.1,0.2,0.3,2\n'
    result = parser.read_csv_check_warnings(ParserWarning, 'Length of header', StringIO(data), sep=',', header=None, index_col=False)
    expected = DataFrame({0: [0.5, 0.1], 1: [0.03, 0.2]})
    tm.assert_frame_equal(result, expected)

def test_header_int_do_not_infer_multiindex_names_on_different_line(python_parser_only):
    if False:
        for i in range(10):
            print('nop')
    parser = python_parser_only
    data = StringIO('a\na,b\nc,d,e\nf,g,h')
    result = parser.read_csv_check_warnings(ParserWarning, 'Length of header', data, engine='python', index_col=False)
    expected = DataFrame({'a': ['a', 'c', 'f']})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('dtype', [{'a': object}, {'a': str, 'b': np.int64, 'c': np.int64}])
def test_no_thousand_convert_with_dot_for_non_numeric_cols(python_parser_only, dtype):
    if False:
        while True:
            i = 10
    parser = python_parser_only
    data = 'a;b;c\n0000.7995;16.000;0\n3.03.001.00514;0;4.000\n4923.600.041;23.000;131'
    result = parser.read_csv(StringIO(data), sep=';', dtype=dtype, thousands='.')
    expected = DataFrame({'a': ['0000.7995', '3.03.001.00514', '4923.600.041'], 'b': [16000, 0, 23000], 'c': [0, 4000, 131]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('dtype,expected', [({'a': str, 'b': np.float64, 'c': np.int64}, DataFrame({'b': [16000.1, 0, 23000], 'c': [0, 4001, 131]})), (str, DataFrame({'b': ['16,000.1', '0', '23,000'], 'c': ['0', '4,001', '131']}))])
def test_no_thousand_convert_for_non_numeric_cols(python_parser_only, dtype, expected):
    if False:
        print('Hello World!')
    parser = python_parser_only
    data = 'a;b;c\n0000,7995;16,000.1;0\n3,03,001,00514;0;4,001\n4923,600,041;23,000;131\n'
    result = parser.read_csv(StringIO(data), sep=';', dtype=dtype, thousands=',')
    expected.insert(0, 'a', ['0000,7995', '3,03,001,00514', '4923,600,041'])
    tm.assert_frame_equal(result, expected)
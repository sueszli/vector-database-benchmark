"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from io import StringIO
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.errors import DtypeWarning
from pandas import DataFrame, concat
import pandas._testing as tm
xfail_pyarrow = pytest.mark.usefixtures('pyarrow_xfail')
pytestmark = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')

@xfail_pyarrow
@pytest.mark.parametrize('index_col', [0, 'index'])
def test_read_chunksize_with_index(all_parsers, index_col):
    if False:
        for i in range(10):
            print('nop')
    parser = all_parsers
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    expected = DataFrame([['foo', 2, 3, 4, 5], ['bar', 7, 8, 9, 10], ['baz', 12, 13, 14, 15], ['qux', 12, 13, 14, 15], ['foo2', 12, 13, 14, 15], ['bar2', 12, 13, 14, 15]], columns=['index', 'A', 'B', 'C', 'D'])
    expected = expected.set_index('index')
    with parser.read_csv(StringIO(data), index_col=0, chunksize=2) as reader:
        chunks = list(reader)
    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])

@xfail_pyarrow
@pytest.mark.parametrize('chunksize', [1.3, 'foo', 0])
def test_read_chunksize_bad(all_parsers, chunksize):
    if False:
        i = 10
        return i + 15
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    msg = "'chunksize' must be an integer >=1"
    with pytest.raises(ValueError, match=msg):
        with parser.read_csv(StringIO(data), chunksize=chunksize) as _:
            pass

@xfail_pyarrow
@pytest.mark.parametrize('chunksize', [2, 8])
def test_read_chunksize_and_nrows(all_parsers, chunksize):
    if False:
        while True:
            i = 10
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    kwargs = {'index_col': 0, 'nrows': 5}
    expected = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=chunksize, **kwargs) as reader:
        tm.assert_frame_equal(concat(reader), expected)

@xfail_pyarrow
def test_read_chunksize_and_nrows_changing_size(all_parsers):
    if False:
        for i in range(10):
            print('nop')
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    kwargs = {'index_col': 0, 'nrows': 5}
    expected = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=8, **kwargs) as reader:
        tm.assert_frame_equal(reader.get_chunk(size=2), expected.iloc[:2])
        tm.assert_frame_equal(reader.get_chunk(size=4), expected.iloc[2:5])
        with pytest.raises(StopIteration, match=''):
            reader.get_chunk(size=3)

@xfail_pyarrow
def test_get_chunk_passed_chunksize(all_parsers):
    if False:
        return 10
    parser = all_parsers
    data = 'A,B,C\n1,2,3\n4,5,6\n7,8,9\n1,2,3'
    with parser.read_csv(StringIO(data), chunksize=2) as reader:
        result = reader.get_chunk()
    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
@pytest.mark.parametrize('kwargs', [{}, {'index_col': 0}])
def test_read_chunksize_compat(all_parsers, kwargs):
    if False:
        print('Hello World!')
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=2, **kwargs) as reader:
        tm.assert_frame_equal(concat(reader), result)

@xfail_pyarrow
def test_read_chunksize_jagged_names(all_parsers):
    if False:
        return 10
    parser = all_parsers
    data = '\n'.join(['0'] * 7 + [','.join(['0'] * 10)])
    expected = DataFrame([[0] + [np.nan] * 9] * 7 + [[0] * 10])
    with parser.read_csv(StringIO(data), names=range(10), chunksize=4) as reader:
        result = concat(reader)
    tm.assert_frame_equal(result, expected)

def test_chunk_begins_with_newline_whitespace(all_parsers):
    if False:
        return 10
    parser = all_parsers
    data = '\n hello\nworld\n'
    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame([' hello', 'world'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.slow
def test_chunks_have_consistent_numerical_type(all_parsers, monkeypatch):
    if False:
        print('Hello World!')
    heuristic = 2 ** 3
    parser = all_parsers
    integers = [str(i) for i in range(heuristic - 1)]
    data = 'a\n' + '\n'.join(integers + ['1.0', '2.0'] + integers)
    warn = None
    if parser.engine == 'pyarrow':
        warn = DeprecationWarning
    depr_msg = 'Passing a BlockManager to DataFrame'
    with tm.assert_produces_warning(warn, match=depr_msg, check_stacklevel=False):
        with monkeypatch.context() as m:
            m.setattr(libparsers, 'DEFAULT_BUFFER_HEURISTIC', heuristic)
            result = parser.read_csv(StringIO(data))
    assert type(result.a[0]) is np.float64
    assert result.a.dtype == float

@xfail_pyarrow
def test_warn_if_chunks_have_mismatched_type(all_parsers):
    if False:
        i = 10
        return i + 15
    warning_type = None
    parser = all_parsers
    size = 10000
    if parser.engine == 'c' and parser.low_memory:
        warning_type = DtypeWarning
        size = 499999
    integers = [str(i) for i in range(size)]
    data = 'a\n' + '\n'.join(integers + ['a', 'b'] + integers)
    buf = StringIO(data)
    df = parser.read_csv_check_warnings(warning_type, 'Columns \\(0\\) have mixed types. Specify dtype option on import or set low_memory=False.', buf)
    assert df.a.dtype == object

@xfail_pyarrow
@pytest.mark.parametrize('iterator', [True, False])
def test_empty_with_nrows_chunksize(all_parsers, iterator):
    if False:
        for i in range(10):
            print('nop')
    parser = all_parsers
    expected = DataFrame(columns=['foo', 'bar'])
    nrows = 10
    data = StringIO('foo,bar\n')
    if iterator:
        with parser.read_csv(data, chunksize=nrows) as reader:
            result = next(iter(reader))
    else:
        result = parser.read_csv(data, nrows=nrows)
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
def test_read_csv_memory_growth_chunksize(all_parsers):
    if False:
        i = 10
        return i + 15
    parser = all_parsers
    with tm.ensure_clean() as path:
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(1000):
                f.write(str(i) + '\n')
        with parser.read_csv(path, chunksize=20) as result:
            for _ in result:
                pass

@xfail_pyarrow
def test_chunksize_with_usecols_second_block_shorter(all_parsers):
    if False:
        for i in range(10):
            print('nop')
    parser = all_parsers
    data = '1,2,3,4\n5,6,7,8\n9,10,11\n'
    result_chunks = parser.read_csv(StringIO(data), names=['a', 'b'], chunksize=2, usecols=[0, 1], header=None)
    expected_frames = [DataFrame({'a': [1, 5], 'b': [2, 6]}), DataFrame({'a': [9], 'b': [10]}, index=[2])]
    for (i, result) in enumerate(result_chunks):
        tm.assert_frame_equal(result, expected_frames[i])

@xfail_pyarrow
def test_chunksize_second_block_shorter(all_parsers):
    if False:
        while True:
            i = 10
    parser = all_parsers
    data = 'a,b,c,d\n1,2,3,4\n5,6,7,8\n9,10,11\n'
    result_chunks = parser.read_csv(StringIO(data), chunksize=2)
    expected_frames = [DataFrame({'a': [1, 5], 'b': [2, 6], 'c': [3, 7], 'd': [4, 8]}), DataFrame({'a': [9], 'b': [10], 'c': [11], 'd': [np.nan]}, index=[2])]
    for (i, result) in enumerate(result_chunks):
        tm.assert_frame_equal(result, expected_frames[i])
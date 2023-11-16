"""
Tests that comments are properly handled during parsing
for all of the parsers defined in parsers.py
"""
from io import StringIO
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
xfail_pyarrow = pytest.mark.usefixtures('pyarrow_xfail')

@xfail_pyarrow
@pytest.mark.parametrize('na_values', [None, ['NaN']])
def test_comment(all_parsers, na_values):
    if False:
        return 10
    parser = all_parsers
    data = 'A,B,C\n1,2.,4.#hello world\n5.,NaN,10.0\n'
    expected = DataFrame([[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=['A', 'B', 'C'])
    result = parser.read_csv(StringIO(data), comment='#', na_values=na_values)
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
@pytest.mark.parametrize('read_kwargs', [{}, {'lineterminator': '*'}, {'delim_whitespace': True}])
def test_line_comment(all_parsers, read_kwargs, request):
    if False:
        return 10
    parser = all_parsers
    data = '# empty\nA,B,C\n1,2.,4.#hello world\n#ignore this line\n5.,NaN,10.0\n'
    if read_kwargs.get('delim_whitespace'):
        data = data.replace(',', ' ')
    elif read_kwargs.get('lineterminator'):
        if parser.engine != 'c':
            mark = pytest.mark.xfail(reason='Custom terminator not supported with Python engine')
            request.applymarker(mark)
        data = data.replace('\n', read_kwargs.get('lineterminator'))
    read_kwargs['comment'] = '#'
    result = parser.read_csv(StringIO(data), **read_kwargs)
    expected = DataFrame([[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
def test_comment_skiprows(all_parsers):
    if False:
        i = 10
        return i + 15
    parser = all_parsers
    data = '# empty\nrandom line\n# second empty line\n1,2,3\nA,B,C\n1,2.,4.\n5.,NaN,10.0\n'
    expected = DataFrame([[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=['A', 'B', 'C'])
    result = parser.read_csv(StringIO(data), comment='#', skiprows=4)
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
def test_comment_header(all_parsers):
    if False:
        i = 10
        return i + 15
    parser = all_parsers
    data = '# empty\n# second empty line\n1,2,3\nA,B,C\n1,2.,4.\n5.,NaN,10.0\n'
    expected = DataFrame([[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=['A', 'B', 'C'])
    result = parser.read_csv(StringIO(data), comment='#', header=1)
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
def test_comment_skiprows_header(all_parsers):
    if False:
        for i in range(10):
            print('nop')
    parser = all_parsers
    data = '# empty\n# second empty line\n# third empty line\nX,Y,Z\n1,2,3\nA,B,C\n1,2.,4.\n5.,NaN,10.0\n'
    expected = DataFrame([[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=['A', 'B', 'C'])
    result = parser.read_csv(StringIO(data), comment='#', skiprows=4, header=1)
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
@pytest.mark.parametrize('comment_char', ['#', '~', '&', '^', '*', '@'])
def test_custom_comment_char(all_parsers, comment_char):
    if False:
        i = 10
        return i + 15
    parser = all_parsers
    data = 'a,b,c\n1,2,3#ignore this!\n4,5,6#ignorethistoo'
    result = parser.read_csv(StringIO(data.replace('#', comment_char)), comment=comment_char)
    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
@pytest.mark.parametrize('header', ['infer', None])
def test_comment_first_line(all_parsers, header):
    if False:
        while True:
            i = 10
    parser = all_parsers
    data = '# notes\na,b,c\n# more notes\n1,2,3'
    if header is None:
        expected = DataFrame({0: ['a', '1'], 1: ['b', '2'], 2: ['c', '3']})
    else:
        expected = DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
    result = parser.read_csv(StringIO(data), comment='#', header=header)
    tm.assert_frame_equal(result, expected)

@xfail_pyarrow
def test_comment_char_in_default_value(all_parsers, request):
    if False:
        return 10
    if all_parsers.engine == 'c':
        reason = 'see gh-34002: works on the python engine but not the c engine'
        request.applymarker(pytest.mark.xfail(reason=reason, raises=AssertionError))
    parser = all_parsers
    data = '# this is a comment\ncol1,col2,col3,col4\n1,2,3,4#inline comment\n4,5#,6,10\n7,8,#N/A,11\n'
    result = parser.read_csv(StringIO(data), comment='#', na_values='#N/A')
    expected = DataFrame({'col1': [1, 4, 7], 'col2': [2, 5, 8], 'col3': [3.0, np.nan, np.nan], 'col4': [4.0, np.nan, 11.0]})
    tm.assert_frame_equal(result, expected)
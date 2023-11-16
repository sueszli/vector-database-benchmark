import re
import numpy as np
import pytest
from pandas import DataFrame, Index, MultiIndex, Series, _testing as tm, concat

@pytest.mark.parametrize('other', [None, Series, Index])
def test_str_cat_name(index_or_series, other):
    if False:
        for i in range(10):
            print('nop')
    box = index_or_series
    values = ['a', 'b']
    if other:
        other = other(values)
    else:
        other = values
    result = box(values, name='name').str.cat(other, sep=',')
    assert result.name == 'name'

def test_str_cat(index_or_series):
    if False:
        while True:
            i = 10
    box = index_or_series
    s = box(['a', 'a', 'b', 'b', 'c', np.nan])
    result = s.str.cat()
    expected = 'aabbc'
    assert result == expected
    result = s.str.cat(na_rep='-')
    expected = 'aabbc-'
    assert result == expected
    result = s.str.cat(sep='_', na_rep='NA')
    expected = 'a_a_b_b_c_NA'
    assert result == expected
    t = np.array(['a', np.nan, 'b', 'd', 'foo', np.nan], dtype=object)
    expected = box(['aa', 'a-', 'bb', 'bd', 'cfoo', '--'])
    result = s.str.cat(t, na_rep='-')
    tm.assert_equal(result, expected)
    result = s.str.cat(list(t), na_rep='-')
    tm.assert_equal(result, expected)
    rgx = 'If `others` contains arrays or lists \\(or other list-likes.*'
    z = Series(['1', '2', '3'])
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(z.values)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(list(z))

def test_str_cat_raises_intuitive_error(index_or_series):
    if False:
        print('Hello World!')
    box = index_or_series
    s = box(['a', 'b', 'c', 'd'])
    message = 'Did you mean to supply a `sep` keyword?'
    with pytest.raises(ValueError, match=message):
        s.str.cat('|')
    with pytest.raises(ValueError, match=message):
        s.str.cat('    ')

@pytest.mark.parametrize('sep', ['', None])
@pytest.mark.parametrize('dtype_target', ['object', 'category'])
@pytest.mark.parametrize('dtype_caller', ['object', 'category'])
def test_str_cat_categorical(index_or_series, dtype_caller, dtype_target, sep):
    if False:
        while True:
            i = 10
    box = index_or_series
    s = Index(['a', 'a', 'b', 'a'], dtype=dtype_caller)
    s = s if box == Index else Series(s, index=s)
    t = Index(['b', 'a', 'b', 'c'], dtype=dtype_target)
    expected = Index(['ab', 'aa', 'bb', 'ac'])
    expected = expected if box == Index else Series(expected, index=s)
    result = s.str.cat(t.values, sep=sep)
    tm.assert_equal(result, expected)
    t = Series(t.values, index=s)
    result = s.str.cat(t, sep=sep)
    tm.assert_equal(result, expected)
    result = s.str.cat(t.values, sep=sep)
    tm.assert_equal(result, expected)
    t = Series(t.values, index=t.values)
    expected = Index(['aa', 'aa', 'bb', 'bb', 'aa'])
    expected = expected if box == Index else Series(expected, index=expected.str[:1])
    result = s.str.cat(t, sep=sep)
    tm.assert_equal(result, expected)

@pytest.mark.parametrize('data', [[1, 2, 3], [0.1, 0.2, 0.3], [1, 2, 'b']], ids=['integers', 'floats', 'mixed'])
@pytest.mark.parametrize('box', [Series, Index, list, lambda x: np.array(x, dtype=object)], ids=['Series', 'Index', 'list', 'np.array'])
def test_str_cat_wrong_dtype_raises(box, data):
    if False:
        print('Hello World!')
    s = Series(['a', 'b', 'c'])
    t = box(data)
    msg = 'Concatenation requires list-likes containing only strings.*'
    with pytest.raises(TypeError, match=msg):
        s.str.cat(t, join='outer', na_rep='-')

def test_str_cat_mixed_inputs(index_or_series):
    if False:
        while True:
            i = 10
    box = index_or_series
    s = Index(['a', 'b', 'c', 'd'])
    s = s if box == Index else Series(s, index=s)
    t = Series(['A', 'B', 'C', 'D'], index=s.values)
    d = concat([t, Series(s, index=s)], axis=1)
    expected = Index(['aAa', 'bBb', 'cCc', 'dDd'])
    expected = expected if box == Index else Series(expected.values, index=s.values)
    result = s.str.cat(d)
    tm.assert_equal(result, expected)
    result = s.str.cat(d.values)
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s])
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s.values])
    tm.assert_equal(result, expected)
    t.index = ['b', 'c', 'd', 'a']
    expected = box(['aDa', 'bAb', 'cBc', 'dCd'])
    expected = expected if box == Index else Series(expected.values, index=s.values)
    result = s.str.cat([t, s])
    tm.assert_equal(result, expected)
    result = s.str.cat([t, s.values])
    tm.assert_equal(result, expected)
    d.index = ['b', 'c', 'd', 'a']
    expected = box(['aDd', 'bAa', 'cBb', 'dCc'])
    expected = expected if box == Index else Series(expected.values, index=s.values)
    result = s.str.cat(d)
    tm.assert_equal(result, expected)
    rgx = 'If `others` contains arrays or lists \\(or other list-likes.*'
    z = Series(['1', '2', '3'])
    e = concat([z, z], axis=1)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(e.values)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([z.values, s.values])
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([z.values, s])
    rgx = 'others must be Series, Index, DataFrame,.*'
    u = Series(['a', np.nan, 'c', None])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, 'u'])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, d])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, d.values])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, [u, d]])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(set(u))
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, set(u)])
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(1)
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(iter([t.values, list(s)]))

@pytest.mark.parametrize('join', ['left', 'outer', 'inner', 'right'])
def test_str_cat_align_indexed(index_or_series, join):
    if False:
        for i in range(10):
            print('nop')
    box = index_or_series
    s = Series(['a', 'b', 'c', 'd'], index=['a', 'b', 'c', 'd'])
    t = Series(['D', 'A', 'E', 'B'], index=['d', 'a', 'e', 'b'])
    (sa, ta) = s.align(t, join=join)
    expected = sa.str.cat(ta, na_rep='-')
    if box == Index:
        s = Index(s)
        sa = Index(sa)
        expected = Index(expected)
    result = s.str.cat(t, join=join, na_rep='-')
    tm.assert_equal(result, expected)

@pytest.mark.parametrize('join', ['left', 'outer', 'inner', 'right'])
def test_str_cat_align_mixed_inputs(join):
    if False:
        for i in range(10):
            print('nop')
    s = Series(['a', 'b', 'c', 'd'])
    t = Series(['d', 'a', 'e', 'b'], index=[3, 0, 4, 1])
    d = concat([t, t], axis=1)
    expected_outer = Series(['aaa', 'bbb', 'c--', 'ddd', '-ee'])
    expected = expected_outer.loc[s.index.join(t.index, how=join)]
    result = s.str.cat([t, t], join=join, na_rep='-')
    tm.assert_series_equal(result, expected)
    result = s.str.cat(d, join=join, na_rep='-')
    tm.assert_series_equal(result, expected)
    u = np.array(['A', 'B', 'C', 'D'])
    expected_outer = Series(['aaA', 'bbB', 'c-C', 'ddD', '-e-'])
    rhs_idx = t.index.intersection(s.index) if join == 'inner' else t.index.union(s.index) if join == 'outer' else t.index.append(s.index.difference(t.index))
    expected = expected_outer.loc[s.index.join(rhs_idx, how=join)]
    result = s.str.cat([t, u], join=join, na_rep='-')
    tm.assert_series_equal(result, expected)
    with pytest.raises(TypeError, match='others must be Series,.*'):
        s.str.cat([t, list(u)], join=join)
    rgx = 'If `others` contains arrays or lists \\(or other list-likes.*'
    z = Series(['1', '2', '3']).values
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(z, join=join)
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([t, z], join=join)

def test_str_cat_all_na(index_or_series, index_or_series2):
    if False:
        print('Hello World!')
    box = index_or_series
    other = index_or_series2
    s = Index(['a', 'b', 'c', 'd'])
    s = s if box == Index else Series(s, index=s)
    t = other([np.nan] * 4, dtype=object)
    t = t if other == Index else Series(t, index=s)
    if box == Series:
        expected = Series([np.nan] * 4, index=s.index, dtype=object)
    else:
        expected = Index([np.nan] * 4, dtype=object)
    result = s.str.cat(t, join='left')
    tm.assert_equal(result, expected)
    if other == Series:
        expected = Series([np.nan] * 4, dtype=object, index=t.index)
        result = t.str.cat(s, join='left')
        tm.assert_series_equal(result, expected)

def test_str_cat_special_cases():
    if False:
        while True:
            i = 10
    s = Series(['a', 'b', 'c', 'd'])
    t = Series(['d', 'a', 'e', 'b'], index=[3, 0, 4, 1])
    expected = Series(['aaa', 'bbb', 'c-c', 'ddd', '-e-'])
    result = s.str.cat(iter([t, s.values]), join='outer', na_rep='-')
    tm.assert_series_equal(result, expected)
    expected = Series(['aa-', 'd-d'], index=[0, 3])
    result = s.str.cat([t.loc[[0]], t.loc[[3]]], join='right', na_rep='-')
    tm.assert_series_equal(result, expected)

def test_cat_on_filtered_index():
    if False:
        while True:
            i = 10
    df = DataFrame(index=MultiIndex.from_product([[2011, 2012], [1, 2, 3]], names=['year', 'month']))
    df = df.reset_index()
    df = df[df.month > 1]
    str_year = df.year.astype('str')
    str_month = df.month.astype('str')
    str_both = str_year.str.cat(str_month, sep=' ')
    assert str_both.loc[1] == '2011 2'
    str_multiple = str_year.str.cat([str_month, str_month], sep=' ')
    assert str_multiple.loc[1] == '2011 2 2'

@pytest.mark.parametrize('klass', [tuple, list, np.array, Series, Index])
def test_cat_different_classes(klass):
    if False:
        while True:
            i = 10
    s = Series(['a', 'b', 'c'])
    result = s.str.cat(klass(['x', 'y', 'z']))
    expected = Series(['ax', 'by', 'cz'])
    tm.assert_series_equal(result, expected)

def test_cat_on_series_dot_str():
    if False:
        while True:
            i = 10
    ps = Series(['AbC', 'de', 'FGHI', 'j', 'kLLLm'])
    message = re.escape('others must be Series, Index, DataFrame, np.ndarray or list-like (either containing only strings or containing only objects of type Series/Index/np.ndarray[1-dim])')
    with pytest.raises(TypeError, match=message):
        ps.str.cat(others=ps.str)
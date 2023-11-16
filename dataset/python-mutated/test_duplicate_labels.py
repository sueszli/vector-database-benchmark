"""Tests dealing with the NDFrame.allows_duplicates."""
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
not_implemented = pytest.mark.xfail(reason='Not implemented.')

class TestPreserves:

    @pytest.mark.parametrize('cls, data', [(pd.Series, np.array([])), (pd.Series, [1, 2]), (pd.DataFrame, {}), (pd.DataFrame, {'A': [1, 2]})])
    def test_construction_ok(self, cls, data):
        if False:
            i = 10
            return i + 15
        result = cls(data)
        assert result.flags.allows_duplicate_labels is True
        result = cls(data).set_flags(allows_duplicate_labels=False)
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize('func', [operator.itemgetter(['a']), operator.methodcaller('add', 1), operator.methodcaller('rename', str.upper), operator.methodcaller('rename', 'name'), operator.methodcaller('abs'), np.abs])
    def test_preserved_series(self, func):
        if False:
            return 10
        s = pd.Series([0, 1], index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        assert func(s).flags.allows_duplicate_labels is False

    @pytest.mark.parametrize('other', [pd.Series(0, index=['a', 'b', 'c']), pd.Series(0, index=['a', 'b'])])
    @not_implemented
    def test_align(self, other):
        if False:
            for i in range(10):
                print('nop')
        s = pd.Series([0, 1], index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        (a, b) = s.align(other)
        assert a.flags.allows_duplicate_labels is False
        assert b.flags.allows_duplicate_labels is False

    def test_preserved_frame(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        assert df.loc[['a']].flags.allows_duplicate_labels is False
        assert df.loc[:, ['A', 'B']].flags.allows_duplicate_labels is False

    def test_to_frame(self):
        if False:
            return 10
        ser = pd.Series(dtype=float).set_flags(allows_duplicate_labels=False)
        assert ser.to_frame().flags.allows_duplicate_labels is False

    @pytest.mark.parametrize('func', ['add', 'sub'])
    @pytest.mark.parametrize('frame', [False, True])
    @pytest.mark.parametrize('other', [1, pd.Series([1, 2], name='A')])
    def test_binops(self, func, other, frame):
        if False:
            i = 10
            return i + 15
        df = pd.Series([1, 2], name='A', index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        if frame:
            df = df.to_frame()
        if isinstance(other, pd.Series) and frame:
            other = other.to_frame()
        func = operator.methodcaller(func, other)
        assert df.flags.allows_duplicate_labels is False
        assert func(df).flags.allows_duplicate_labels is False

    def test_preserve_getitem(self):
        if False:
            return 10
        df = pd.DataFrame({'A': [1, 2]}).set_flags(allows_duplicate_labels=False)
        assert df[['A']].flags.allows_duplicate_labels is False
        assert df['A'].flags.allows_duplicate_labels is False
        assert df.loc[0].flags.allows_duplicate_labels is False
        assert df.loc[[0]].flags.allows_duplicate_labels is False
        assert df.loc[0, ['A']].flags.allows_duplicate_labels is False

    def test_ndframe_getitem_caching_issue(self, request, using_copy_on_write, warn_copy_on_write):
        if False:
            print('Hello World!')
        if not (using_copy_on_write or warn_copy_on_write):
            request.applymarker(pytest.mark.xfail(reason='Unclear behavior.'))
        df = pd.DataFrame({'A': [0]}).set_flags(allows_duplicate_labels=False)
        assert df['A'].flags.allows_duplicate_labels is False
        df.flags.allows_duplicate_labels = True
        assert df['A'].flags.allows_duplicate_labels is True

    @pytest.mark.parametrize('objs, kwargs', [([pd.Series(1, index=['a', 'b']), pd.Series(2, index=['c', 'd'])], {}), ([pd.Series(1, index=['a', 'b']), pd.Series(2, index=['a', 'b'])], {'ignore_index': True}), ([pd.Series(1, index=['a', 'b']), pd.Series(2, index=['a', 'b'])], {'axis': 1}), ([pd.DataFrame({'A': [1, 2]}, index=['a', 'b']), pd.DataFrame({'A': [1, 2]}, index=['c', 'd'])], {}), ([pd.DataFrame({'A': [1, 2]}, index=['a', 'b']), pd.DataFrame({'A': [1, 2]}, index=['a', 'b'])], {'ignore_index': True}), ([pd.DataFrame({'A': [1, 2]}, index=['a', 'b']), pd.DataFrame({'B': [1, 2]}, index=['a', 'b'])], {'axis': 1}), ([pd.DataFrame({'A': [1, 2]}, index=['a', 'b']), pd.Series([1, 2], index=['a', 'b'], name='B')], {'axis': 1})])
    def test_concat(self, objs, kwargs):
        if False:
            print('Hello World!')
        objs = [x.set_flags(allows_duplicate_labels=False) for x in objs]
        result = pd.concat(objs, **kwargs)
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize('left, right, expected', [pytest.param(pd.DataFrame({'A': [0, 1]}, index=['a', 'b']).set_flags(allows_duplicate_labels=False), pd.DataFrame({'B': [0, 1]}, index=['a', 'd']).set_flags(allows_duplicate_labels=False), False, marks=not_implemented), pytest.param(pd.DataFrame({'A': [0, 1]}, index=['a', 'b']).set_flags(allows_duplicate_labels=False), pd.DataFrame({'B': [0, 1]}, index=['a', 'd']), False, marks=not_implemented), (pd.DataFrame({'A': [0, 1]}, index=['a', 'b']), pd.DataFrame({'B': [0, 1]}, index=['a', 'd']), True)])
    def test_merge(self, left, right, expected):
        if False:
            i = 10
            return i + 15
        result = pd.merge(left, right, left_index=True, right_index=True)
        assert result.flags.allows_duplicate_labels is expected

    @not_implemented
    def test_groupby(self):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'A': [1, 2, 3]}).set_flags(allows_duplicate_labels=False)
        result = df.groupby([0, 0, 1]).agg('count')
        assert result.flags.allows_duplicate_labels is False

    @pytest.mark.parametrize('frame', [True, False])
    @not_implemented
    def test_window(self, frame):
        if False:
            for i in range(10):
                print('nop')
        df = pd.Series(1, index=pd.date_range('2000', periods=12), name='A', allows_duplicate_labels=False)
        if frame:
            df = df.to_frame()
        assert df.rolling(3).mean().flags.allows_duplicate_labels is False
        assert df.ewm(3).mean().flags.allows_duplicate_labels is False
        assert df.expanding(3).mean().flags.allows_duplicate_labels is False

class TestRaises:

    @pytest.mark.parametrize('cls, axes', [(pd.Series, {'index': ['a', 'a'], 'dtype': float}), (pd.DataFrame, {'index': ['a', 'a']}), (pd.DataFrame, {'index': ['a', 'a'], 'columns': ['b', 'b']}), (pd.DataFrame, {'columns': ['b', 'b']})])
    def test_set_flags_with_duplicates(self, cls, axes):
        if False:
            return 10
        result = cls(**axes)
        assert result.flags.allows_duplicate_labels is True
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            cls(**axes).set_flags(allows_duplicate_labels=False)

    @pytest.mark.parametrize('data', [pd.Series(index=[0, 0], dtype=float), pd.DataFrame(index=[0, 0]), pd.DataFrame(columns=[0, 0])])
    def test_setting_allows_duplicate_labels_raises(self, data):
        if False:
            i = 10
            return i + 15
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            data.flags.allows_duplicate_labels = False
        assert data.flags.allows_duplicate_labels is True

    def test_series_raises(self):
        if False:
            i = 10
            return i + 15
        a = pd.Series(0, index=['a', 'b'])
        b = pd.Series([0, 1], index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.concat([a, b])

    @pytest.mark.parametrize('getter, target', [(operator.itemgetter(['A', 'A']), None), (operator.itemgetter(['a', 'a']), 'loc'), pytest.param(operator.itemgetter(('a', ['A', 'A'])), 'loc'), (operator.itemgetter((['a', 'a'], 'A')), 'loc'), (operator.itemgetter([0, 0]), 'iloc'), pytest.param(operator.itemgetter((0, [0, 0])), 'iloc'), pytest.param(operator.itemgetter(([0, 0], 0)), 'iloc')])
    def test_getitem_raises(self, getter, target):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['a', 'b']).set_flags(allows_duplicate_labels=False)
        if target:
            target = getattr(df, target)
        else:
            target = df
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            getter(target)

    @pytest.mark.parametrize('objs, kwargs', [([pd.Series(1, index=[0, 1], name='a'), pd.Series(2, index=[0, 1], name='a')], {'axis': 1})])
    def test_concat_raises(self, objs, kwargs):
        if False:
            print('Hello World!')
        objs = [x.set_flags(allows_duplicate_labels=False) for x in objs]
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.concat(objs, **kwargs)

    @not_implemented
    def test_merge_raises(self):
        if False:
            while True:
                i = 10
        a = pd.DataFrame({'A': [0, 1, 2]}, index=['a', 'b', 'c']).set_flags(allows_duplicate_labels=False)
        b = pd.DataFrame({'B': [0, 1, 2]}, index=['a', 'b', 'b'])
        msg = 'Index has duplicates.'
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.merge(a, b, left_index=True, right_index=True)

@pytest.mark.parametrize('idx', [pd.Index([1, 1]), pd.Index(['a', 'a']), pd.Index([1.1, 1.1]), pd.PeriodIndex([pd.Period('2000', 'D')] * 2), pd.DatetimeIndex([pd.Timestamp('2000')] * 2), pd.TimedeltaIndex([pd.Timedelta('1D')] * 2), pd.CategoricalIndex(['a', 'a']), pd.IntervalIndex([pd.Interval(0, 1)] * 2), pd.MultiIndex.from_tuples([('a', 1), ('a', 1)])], ids=lambda x: type(x).__name__)
def test_raises_basic(idx):
    if False:
        print('Hello World!')
    msg = 'Index has duplicates.'
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.Series(1, index=idx).set_flags(allows_duplicate_labels=False)
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.DataFrame({'A': [1, 1]}, index=idx).set_flags(allows_duplicate_labels=False)
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.DataFrame([[1, 2]], columns=idx).set_flags(allows_duplicate_labels=False)

def test_format_duplicate_labels_message():
    if False:
        print('Hello World!')
    idx = pd.Index(['a', 'b', 'a', 'b', 'c'])
    result = idx._format_duplicate_message()
    expected = pd.DataFrame({'positions': [[0, 2], [1, 3]]}, index=pd.Index(['a', 'b'], name='label'))
    tm.assert_frame_equal(result, expected)

def test_format_duplicate_labels_message_multi():
    if False:
        i = 10
        return i + 15
    idx = pd.MultiIndex.from_product([['A'], ['a', 'b', 'a', 'b', 'c']])
    result = idx._format_duplicate_message()
    expected = pd.DataFrame({'positions': [[0, 2], [1, 3]]}, index=pd.MultiIndex.from_product([['A'], ['a', 'b']]))
    tm.assert_frame_equal(result, expected)

def test_dataframe_insert_raises():
    if False:
        i = 10
        return i + 15
    df = pd.DataFrame({'A': [1, 2]}).set_flags(allows_duplicate_labels=False)
    msg = 'Cannot specify'
    with pytest.raises(ValueError, match=msg):
        df.insert(0, 'A', [3, 4], allow_duplicates=True)

@pytest.mark.parametrize('method, frame_only', [(operator.methodcaller('set_index', 'A', inplace=True), True), (operator.methodcaller('reset_index', inplace=True), True), (operator.methodcaller('rename', lambda x: x, inplace=True), False)])
def test_inplace_raises(method, frame_only):
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame({'A': [0, 0], 'B': [1, 2]}).set_flags(allows_duplicate_labels=False)
    s = df['A']
    s.flags.allows_duplicate_labels = False
    msg = 'Cannot specify'
    with pytest.raises(ValueError, match=msg):
        method(df)
    if not frame_only:
        with pytest.raises(ValueError, match=msg):
            method(s)

def test_pickle():
    if False:
        i = 10
        return i + 15
    a = pd.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    b = tm.round_trip_pickle(a)
    tm.assert_series_equal(a, b)
    a = pd.DataFrame({'A': []}).set_flags(allows_duplicate_labels=False)
    b = tm.round_trip_pickle(a)
    tm.assert_frame_equal(a, b)
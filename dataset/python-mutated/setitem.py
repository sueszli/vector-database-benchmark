import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm

class BaseSetitemTests:

    @pytest.fixture(params=[lambda x: x.index, lambda x: list(x.index), lambda x: slice(None), lambda x: slice(0, len(x)), lambda x: range(len(x)), lambda x: list(range(len(x))), lambda x: np.ones(len(x), dtype=bool)], ids=['index', 'list[index]', 'null_slice', 'full_slice', 'range', 'list(range)', 'mask'])
    def full_indexer(self, request):
        if False:
            print('Hello World!')
        '\n        Fixture for an indexer to pass to obj.loc to get/set the full length of the\n        object.\n\n        In some cases, assumes that obj.index is the default RangeIndex.\n        '
        return request.param

    @pytest.fixture(autouse=True)
    def skip_if_immutable(self, dtype, request):
        if False:
            return 10
        if dtype._is_immutable:
            node = request.node
            if node.name.split('[')[0] == 'test_is_immutable':
                return
            pytest.skip('__setitem__ test not applicable with immutable dtype')

    def test_is_immutable(self, data):
        if False:
            while True:
                i = 10
        if data.dtype._is_immutable:
            with pytest.raises(TypeError):
                data[0] = data[0]
        else:
            data[0] = data[1]
            assert data[0] == data[1]

    def test_setitem_scalar_series(self, data, box_in_series):
        if False:
            i = 10
            return i + 15
        if box_in_series:
            data = pd.Series(data)
        data[0] = data[1]
        assert data[0] == data[1]

    def test_setitem_sequence(self, data, box_in_series):
        if False:
            while True:
                i = 10
        if box_in_series:
            data = pd.Series(data)
        original = data.copy()
        data[[0, 1]] = [data[1], data[0]]
        assert data[0] == original[1]
        assert data[1] == original[0]

    def test_setitem_sequence_mismatched_length_raises(self, data, as_array):
        if False:
            i = 10
            return i + 15
        ser = pd.Series(data)
        original = ser.copy()
        value = [data[0]]
        if as_array:
            value = data._from_sequence(value)
        xpr = 'cannot set using a {} indexer with a different length'
        with pytest.raises(ValueError, match=xpr.format('list-like')):
            ser[[0, 1]] = value
        tm.assert_series_equal(ser, original)
        with pytest.raises(ValueError, match=xpr.format('slice')):
            ser[slice(3)] = value
        tm.assert_series_equal(ser, original)

    def test_setitem_empty_indexer(self, data, box_in_series):
        if False:
            for i in range(10):
                print('nop')
        if box_in_series:
            data = pd.Series(data)
        original = data.copy()
        data[np.array([], dtype=int)] = []
        tm.assert_equal(data, original)

    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        if False:
            for i in range(10):
                print('nop')
        if box_in_series:
            data = pd.Series(data)
        data[[0, 1]] = data[2]
        assert data[0] == data[2]
        assert data[1] == data[2]

    @pytest.mark.parametrize('setter', ['loc', 'iloc'])
    def test_setitem_scalar(self, data, setter):
        if False:
            print('Hello World!')
        arr = pd.Series(data)
        setter = getattr(arr, setter)
        setter[0] = data[1]
        assert arr[0] == data[1]

    def test_setitem_loc_scalar_mixed(self, data):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'A': np.arange(len(data)), 'B': data})
        df.loc[0, 'B'] = data[1]
        assert df.loc[0, 'B'] == data[1]

    def test_setitem_loc_scalar_single(self, data):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'B': data})
        df.loc[10, 'B'] = data[1]
        assert df.loc[10, 'B'] == data[1]

    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'A': data, 'B': data})
        df.loc[10, 'B'] = data[1]
        assert df.loc[10, 'B'] == data[1]

    def test_setitem_iloc_scalar_mixed(self, data):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'A': np.arange(len(data)), 'B': data})
        df.iloc[0, 1] = data[1]
        assert df.loc[0, 'B'] == data[1]

    def test_setitem_iloc_scalar_single(self, data):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'B': data})
        df.iloc[10, 0] = data[1]
        assert df.loc[10, 'B'] == data[1]

    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'A': data, 'B': data})
        df.iloc[10, 1] = data[1]
        assert df.loc[10, 'B'] == data[1]

    @pytest.mark.parametrize('mask', [np.array([True, True, True, False, False]), pd.array([True, True, True, False, False], dtype='boolean'), pd.array([True, True, True, pd.NA, pd.NA], dtype='boolean')], ids=['numpy-array', 'boolean-array', 'boolean-array-na'])
    def test_setitem_mask(self, data, mask, box_in_series):
        if False:
            return 10
        arr = data[:5].copy()
        expected = arr.take([0, 0, 0, 3, 4])
        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)
        arr[mask] = data[0]
        tm.assert_equal(expected, arr)

    def test_setitem_mask_raises(self, data, box_in_series):
        if False:
            for i in range(10):
                print('nop')
        mask = np.array([True, False])
        if box_in_series:
            data = pd.Series(data)
        with pytest.raises(IndexError, match='wrong length'):
            data[mask] = data[0]
        mask = pd.array(mask, dtype='boolean')
        with pytest.raises(IndexError, match='wrong length'):
            data[mask] = data[0]

    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
        if False:
            print('Hello World!')
        mask = pd.array(np.zeros(data.shape, dtype='bool'), dtype='boolean')
        mask[:3] = True
        mask[3:5] = pd.NA
        if box_in_series:
            data = pd.Series(data)
        data[mask] = data[0]
        assert (data[:3] == data[0]).all()

    @pytest.mark.parametrize('idx', [[0, 1, 2], pd.array([0, 1, 2], dtype='Int64'), np.array([0, 1, 2])], ids=['list', 'integer-array', 'numpy-array'])
    def test_setitem_integer_array(self, data, idx, box_in_series):
        if False:
            print('Hello World!')
        arr = data[:5].copy()
        expected = data.take([0, 0, 0, 3, 4])
        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)
        arr[idx] = arr[0]
        tm.assert_equal(arr, expected)

    @pytest.mark.parametrize('idx, box_in_series', [([0, 1, 2, pd.NA], False), pytest.param([0, 1, 2, pd.NA], True, marks=pytest.mark.xfail(reason='GH-31948')), (pd.array([0, 1, 2, pd.NA], dtype='Int64'), False), (pd.array([0, 1, 2, pd.NA], dtype='Int64'), False)], ids=['list-False', 'list-True', 'integer-array-False', 'integer-array-True'])
    def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
        if False:
            return 10
        arr = data.copy()
        if box_in_series:
            arr = pd.Series(data, index=[chr(100 + i) for i in range(len(data))])
        msg = 'Cannot index with an integer indexer containing NA values'
        with pytest.raises(ValueError, match=msg):
            arr[idx] = arr[0]

    @pytest.mark.parametrize('as_callable', [True, False])
    @pytest.mark.parametrize('setter', ['loc', None])
    def test_setitem_mask_aligned(self, data, as_callable, setter):
        if False:
            for i in range(10):
                print('nop')
        ser = pd.Series(data)
        mask = np.zeros(len(data), dtype=bool)
        mask[:2] = True
        if as_callable:
            mask2 = lambda x: mask
        else:
            mask2 = mask
        if setter:
            target = getattr(ser, setter)
        else:
            target = ser
        target[mask2] = data[5:7]
        ser[mask2] = data[5:7]
        assert ser[0] == data[5]
        assert ser[1] == data[6]

    @pytest.mark.parametrize('setter', ['loc', None])
    def test_setitem_mask_broadcast(self, data, setter):
        if False:
            print('Hello World!')
        ser = pd.Series(data)
        mask = np.zeros(len(data), dtype=bool)
        mask[:2] = True
        if setter:
            target = getattr(ser, setter)
        else:
            target = ser
        target[mask] = data[10]
        assert ser[0] == data[10]
        assert ser[1] == data[10]

    def test_setitem_expand_columns(self, data):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'A': data})
        result = df.copy()
        result['B'] = 1
        expected = pd.DataFrame({'A': data, 'B': [1] * len(data)})
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        result.loc[:, 'B'] = 1
        tm.assert_frame_equal(result, expected)
        result['B'] = data
        expected = pd.DataFrame({'A': data, 'B': data})
        tm.assert_frame_equal(result, expected)

    def test_setitem_expand_with_extension(self, data):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'A': [1] * len(data)})
        result = df.copy()
        result['B'] = data
        expected = pd.DataFrame({'A': [1] * len(data), 'B': data})
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        result.loc[:, 'B'] = data
        tm.assert_frame_equal(result, expected)

    def test_setitem_frame_invalid_length(self, data):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'A': [1] * len(data)})
        xpr = f'Length of values \\({len(data[:5])}\\) does not match length of index \\({len(df)}\\)'
        with pytest.raises(ValueError, match=xpr):
            df['B'] = data[:5]

    def test_setitem_tuple_index(self, data):
        if False:
            while True:
                i = 10
        ser = pd.Series(data[:2], index=[(0, 0), (0, 1)])
        expected = pd.Series(data.take([1, 1]), index=ser.index)
        ser[0, 0] = data[1]
        tm.assert_series_equal(ser, expected)

    def test_setitem_slice(self, data, box_in_series):
        if False:
            while True:
                i = 10
        arr = data[:5].copy()
        expected = data.take([0, 0, 0, 3, 4])
        if box_in_series:
            arr = pd.Series(arr)
            expected = pd.Series(expected)
        arr[:3] = data[0]
        tm.assert_equal(arr, expected)

    def test_setitem_loc_iloc_slice(self, data):
        if False:
            for i in range(10):
                print('nop')
        arr = data[:5].copy()
        s = pd.Series(arr, index=['a', 'b', 'c', 'd', 'e'])
        expected = pd.Series(data.take([0, 0, 0, 3, 4]), index=s.index)
        result = s.copy()
        result.iloc[:3] = data[0]
        tm.assert_equal(result, expected)
        result = s.copy()
        result.loc[:'c'] = data[0]
        tm.assert_equal(result, expected)

    def test_setitem_slice_mismatch_length_raises(self, data):
        if False:
            for i in range(10):
                print('nop')
        arr = data[:5]
        with pytest.raises(ValueError):
            arr[:1] = arr[:2]

    def test_setitem_slice_array(self, data):
        if False:
            i = 10
            return i + 15
        arr = data[:5].copy()
        arr[:5] = data[-5:]
        tm.assert_extension_array_equal(arr, data[-5:])

    def test_setitem_scalar_key_sequence_raise(self, data):
        if False:
            while True:
                i = 10
        arr = data[:5].copy()
        with pytest.raises(ValueError):
            arr[0] = arr[[0, 1]]

    def test_setitem_preserves_views(self, data):
        if False:
            i = 10
            return i + 15
        view1 = data.view()
        view2 = data[:]
        data[0] = data[1]
        assert view1[0] == data[1]
        assert view2[0] == data[1]

    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer):
        if False:
            for i in range(10):
                print('nop')
        df = expected = pd.DataFrame({'data': pd.Series(data)})
        result = pd.DataFrame(index=df.index)
        key = full_indexer(df)
        result.loc[key, 'data'] = df['data']
        tm.assert_frame_equal(result, expected)

    def test_setitem_with_expansion_row(self, data):
        if False:
            print('Hello World!')
        na_value = data.dtype.na_value
        df = pd.DataFrame({'data': data[:1]})
        df.loc[1, 'data'] = data[1]
        expected = pd.DataFrame({'data': data[:2]})
        tm.assert_frame_equal(df, expected)
        df.loc[2, 'data'] = na_value
        expected = pd.DataFrame({'data': pd.Series([data[0], data[1], na_value], dtype=data.dtype)})
        tm.assert_frame_equal(df, expected)

    def test_setitem_series(self, data, full_indexer):
        if False:
            while True:
                i = 10
        ser = pd.Series(data, name='data')
        result = pd.Series(index=ser.index, dtype=object, name='data')
        key = full_indexer(ser)
        result.loc[key] = ser
        expected = pd.Series(data.astype(object), index=ser.index, name='data', dtype=object)
        tm.assert_series_equal(result, expected)

    def test_setitem_frame_2d_values(self, data):
        if False:
            i = 10
            return i + 15
        df = pd.DataFrame({'A': data})
        using_array_manager = isinstance(df._mgr, pd.core.internals.ArrayManager)
        using_copy_on_write = pd.options.mode.copy_on_write
        blk_data = df._mgr.arrays[0]
        orig = df.copy()
        df.iloc[:] = df
        tm.assert_frame_equal(df, orig)
        df.iloc[:-1] = df.iloc[:-1].copy()
        tm.assert_frame_equal(df, orig)
        df.iloc[:] = df.values
        tm.assert_frame_equal(df, orig)
        if not using_array_manager and (not using_copy_on_write):
            assert df._mgr.arrays[0] is blk_data
        df.iloc[:-1] = df.values[:-1]
        tm.assert_frame_equal(df, orig)

    def test_delitem_series(self, data):
        if False:
            return 10
        ser = pd.Series(data, name='data')
        taker = np.arange(len(ser))
        taker = np.delete(taker, 1)
        expected = ser[taker]
        del ser[1]
        tm.assert_series_equal(ser, expected)

    def test_setitem_invalid(self, data, invalid_scalar):
        if False:
            print('Hello World!')
        msg = ''
        with pytest.raises((ValueError, TypeError), match=msg):
            data[0] = invalid_scalar
        with pytest.raises((ValueError, TypeError), match=msg):
            data[:] = invalid_scalar

    def test_setitem_2d_values(self, data):
        if False:
            for i in range(10):
                print('nop')
        original = data.copy()
        df = pd.DataFrame({'a': data, 'b': data})
        df.loc[[0, 1], :] = df.loc[[1, 0], :].values
        assert (df.loc[0, :] == original[1]).all()
        assert (df.loc[1, :] == original[0]).all()
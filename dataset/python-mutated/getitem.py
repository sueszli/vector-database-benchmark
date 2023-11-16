import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm

class BaseGetitemTests:
    """Tests for ExtensionArray.__getitem__."""

    def test_iloc_series(self, data):
        if False:
            return 10
        ser = pd.Series(data)
        result = ser.iloc[:4]
        expected = pd.Series(data[:4])
        tm.assert_series_equal(result, expected)
        result = ser.iloc[[0, 1, 2, 3]]
        tm.assert_series_equal(result, expected)

    def test_iloc_frame(self, data):
        if False:
            return 10
        df = pd.DataFrame({'A': data, 'B': np.arange(len(data), dtype='int64')})
        expected = pd.DataFrame({'A': data[:4]})
        result = df.iloc[:4, [0]]
        tm.assert_frame_equal(result, expected)
        result = df.iloc[[0, 1, 2, 3], [0]]
        tm.assert_frame_equal(result, expected)
        expected = pd.Series(data[:4], name='A')
        result = df.iloc[:4, 0]
        tm.assert_series_equal(result, expected)
        result = df.iloc[:4, 0]
        tm.assert_series_equal(result, expected)
        result = df.iloc[:, ::2]
        tm.assert_frame_equal(result, df[['A']])
        result = df[['B', 'A']].iloc[:, ::2]
        tm.assert_frame_equal(result, df[['B']])

    def test_iloc_frame_single_block(self, data):
        if False:
            while True:
                i = 10
        df = pd.DataFrame({'A': data})
        result = df.iloc[:, :]
        tm.assert_frame_equal(result, df)
        result = df.iloc[:, :1]
        tm.assert_frame_equal(result, df)
        result = df.iloc[:, :2]
        tm.assert_frame_equal(result, df)
        result = df.iloc[:, ::2]
        tm.assert_frame_equal(result, df)
        result = df.iloc[:, 1:2]
        tm.assert_frame_equal(result, df.iloc[:, :0])
        result = df.iloc[:, -1:]
        tm.assert_frame_equal(result, df)

    def test_loc_series(self, data):
        if False:
            return 10
        ser = pd.Series(data)
        result = ser.loc[:3]
        expected = pd.Series(data[:4])
        tm.assert_series_equal(result, expected)
        result = ser.loc[[0, 1, 2, 3]]
        tm.assert_series_equal(result, expected)

    def test_loc_frame(self, data):
        if False:
            print('Hello World!')
        df = pd.DataFrame({'A': data, 'B': np.arange(len(data), dtype='int64')})
        expected = pd.DataFrame({'A': data[:4]})
        result = df.loc[:3, ['A']]
        tm.assert_frame_equal(result, expected)
        result = df.loc[[0, 1, 2, 3], ['A']]
        tm.assert_frame_equal(result, expected)
        expected = pd.Series(data[:4], name='A')
        result = df.loc[:3, 'A']
        tm.assert_series_equal(result, expected)
        result = df.loc[:3, 'A']
        tm.assert_series_equal(result, expected)

    def test_loc_iloc_frame_single_dtype(self, data):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'A': data})
        expected = pd.Series([data[2]], index=['A'], name=2, dtype=data.dtype)
        result = df.loc[2]
        tm.assert_series_equal(result, expected)
        expected = pd.Series([data[-1]], index=['A'], name=len(data) - 1, dtype=data.dtype)
        result = df.iloc[-1]
        tm.assert_series_equal(result, expected)

    def test_getitem_scalar(self, data):
        if False:
            return 10
        result = data[0]
        assert isinstance(result, data.dtype.type)
        result = pd.Series(data)[0]
        assert isinstance(result, data.dtype.type)

    def test_getitem_invalid(self, data):
        if False:
            i = 10
            return i + 15
        msg = 'only integers, slices \\(`:`\\), ellipsis \\(`...`\\), numpy.newaxis \\(`None`\\) and integer or boolean arrays are valid indices'
        with pytest.raises(IndexError, match=msg):
            data['foo']
        with pytest.raises(IndexError, match=msg):
            data[2.5]
        ub = len(data)
        msg = '|'.join(['list index out of range', 'index out of bounds', 'Out of bounds access', f'loc must be an integer between -{ub} and {ub}', f'index {ub + 1} is out of bounds for axis 0 with size {ub}', f'index -{ub + 1} is out of bounds for axis 0 with size {ub}'])
        with pytest.raises(IndexError, match=msg):
            data[ub + 1]
        with pytest.raises(IndexError, match=msg):
            data[-ub - 1]

    def test_getitem_scalar_na(self, data_missing, na_cmp):
        if False:
            while True:
                i = 10
        na_value = data_missing.dtype.na_value
        result = data_missing[0]
        assert na_cmp(result, na_value)

    def test_getitem_empty(self, data):
        if False:
            for i in range(10):
                print('nop')
        result = data[[]]
        assert len(result) == 0
        assert isinstance(result, type(data))
        expected = data[np.array([], dtype='int64')]
        tm.assert_extension_array_equal(result, expected)

    def test_getitem_mask(self, data):
        if False:
            return 10
        mask = np.zeros(len(data), dtype=bool)
        result = data[mask]
        assert len(result) == 0
        assert isinstance(result, type(data))
        mask = np.zeros(len(data), dtype=bool)
        result = pd.Series(data)[mask]
        assert len(result) == 0
        assert result.dtype == data.dtype
        mask[0] = True
        result = data[mask]
        assert len(result) == 1
        assert isinstance(result, type(data))
        result = pd.Series(data)[mask]
        assert len(result) == 1
        assert result.dtype == data.dtype

    def test_getitem_mask_raises(self, data):
        if False:
            print('Hello World!')
        mask = np.array([True, False])
        msg = f'Boolean index has wrong length: 2 instead of {len(data)}'
        with pytest.raises(IndexError, match=msg):
            data[mask]
        mask = pd.array(mask, dtype='boolean')
        with pytest.raises(IndexError, match=msg):
            data[mask]

    def test_getitem_boolean_array_mask(self, data):
        if False:
            print('Hello World!')
        mask = pd.array(np.zeros(data.shape, dtype='bool'), dtype='boolean')
        result = data[mask]
        assert len(result) == 0
        assert isinstance(result, type(data))
        result = pd.Series(data)[mask]
        assert len(result) == 0
        assert result.dtype == data.dtype
        mask[:5] = True
        expected = data.take([0, 1, 2, 3, 4])
        result = data[mask]
        tm.assert_extension_array_equal(result, expected)
        expected = pd.Series(expected)
        result = pd.Series(data)[mask]
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean_na_treated_as_false(self, data):
        if False:
            print('Hello World!')
        mask = pd.array(np.zeros(data.shape, dtype='bool'), dtype='boolean')
        mask[:2] = pd.NA
        mask[2:4] = True
        result = data[mask]
        expected = data[mask.fillna(False)]
        tm.assert_extension_array_equal(result, expected)
        s = pd.Series(data)
        result = s[mask]
        expected = s[mask.fillna(False)]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('idx', [[0, 1, 2], pd.array([0, 1, 2], dtype='Int64'), np.array([0, 1, 2])], ids=['list', 'integer-array', 'numpy-array'])
    def test_getitem_integer_array(self, data, idx):
        if False:
            for i in range(10):
                print('nop')
        result = data[idx]
        assert len(result) == 3
        assert isinstance(result, type(data))
        expected = data.take([0, 1, 2])
        tm.assert_extension_array_equal(result, expected)
        expected = pd.Series(expected)
        result = pd.Series(data)[idx]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('idx', [[0, 1, 2, pd.NA], pd.array([0, 1, 2, pd.NA], dtype='Int64')], ids=['list', 'integer-array'])
    def test_getitem_integer_with_missing_raises(self, data, idx):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Cannot index with an integer indexer containing NA values'
        with pytest.raises(ValueError, match=msg):
            data[idx]

    @pytest.mark.xfail(reason='Tries label-based and raises KeyError; in some cases raises when calling np.asarray')
    @pytest.mark.parametrize('idx', [[0, 1, 2, pd.NA], pd.array([0, 1, 2, pd.NA], dtype='Int64')], ids=['list', 'integer-array'])
    def test_getitem_series_integer_with_missing_raises(self, data, idx):
        if False:
            print('Hello World!')
        msg = 'Cannot index with an integer indexer containing NA values'
        ser = pd.Series(data, index=[chr(100 + i) for i in range(len(data))])
        with pytest.raises(ValueError, match=msg):
            ser[idx]

    def test_getitem_slice(self, data):
        if False:
            print('Hello World!')
        result = data[slice(0)]
        assert isinstance(result, type(data))
        result = data[slice(1)]
        assert isinstance(result, type(data))

    def test_getitem_ellipsis_and_slice(self, data):
        if False:
            while True:
                i = 10
        result = data[..., :]
        tm.assert_extension_array_equal(result, data)
        result = data[:, ...]
        tm.assert_extension_array_equal(result, data)
        result = data[..., :3]
        tm.assert_extension_array_equal(result, data[:3])
        result = data[:3, ...]
        tm.assert_extension_array_equal(result, data[:3])
        result = data[..., ::2]
        tm.assert_extension_array_equal(result, data[::2])
        result = data[::2, ...]
        tm.assert_extension_array_equal(result, data[::2])

    def test_get(self, data):
        if False:
            while True:
                i = 10
        s = pd.Series(data, index=[2 * i for i in range(len(data))])
        assert s.get(4) == s.iloc[2]
        result = s.get([4, 6])
        expected = s.iloc[[2, 3]]
        tm.assert_series_equal(result, expected)
        result = s.get(slice(2))
        expected = s.iloc[[0, 1]]
        tm.assert_series_equal(result, expected)
        assert s.get(-1) is None
        assert s.get(s.index.max() + 1) is None
        s = pd.Series(data[:6], index=list('abcdef'))
        assert s.get('c') == s.iloc[2]
        result = s.get(slice('b', 'd'))
        expected = s.iloc[[1, 2, 3]]
        tm.assert_series_equal(result, expected)
        result = s.get('Z')
        assert result is None
        msg = 'Series.__getitem__ treating keys as positions is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert s.get(4) == s.iloc[4]
            assert s.get(-1) == s.iloc[-1]
            assert s.get(len(s)) is None
        s = pd.Series(data)
        with tm.assert_produces_warning(None):
            s2 = s[::2]
        assert s2.get(1) is None

    def test_take_sequence(self, data):
        if False:
            while True:
                i = 10
        result = pd.Series(data)[[0, 1, 3]]
        assert result.iloc[0] == data[0]
        assert result.iloc[1] == data[1]
        assert result.iloc[2] == data[3]

    def test_take(self, data, na_cmp):
        if False:
            return 10
        na_value = data.dtype.na_value
        result = data.take([0, -1])
        assert result.dtype == data.dtype
        assert result[0] == data[0]
        assert result[1] == data[-1]
        result = data.take([0, -1], allow_fill=True, fill_value=na_value)
        assert result[0] == data[0]
        assert na_cmp(result[1], na_value)
        with pytest.raises(IndexError, match='out of bounds'):
            data.take([len(data) + 1])

    def test_take_empty(self, data, na_cmp):
        if False:
            while True:
                i = 10
        na_value = data.dtype.na_value
        empty = data[:0]
        result = empty.take([-1], allow_fill=True)
        assert na_cmp(result[0], na_value)
        msg = 'cannot do a non-empty take from an empty axes|out of bounds'
        with pytest.raises(IndexError, match=msg):
            empty.take([-1])
        with pytest.raises(IndexError, match='cannot do a non-empty take'):
            empty.take([0, 1])

    def test_take_negative(self, data):
        if False:
            return 10
        n = len(data)
        result = data.take([0, -n, n - 1, -1])
        expected = data.take([0, 0, n - 1, n - 1])
        tm.assert_extension_array_equal(result, expected)

    def test_take_non_na_fill_value(self, data_missing):
        if False:
            while True:
                i = 10
        fill_value = data_missing[1]
        na = data_missing[0]
        arr = data_missing._from_sequence([na, fill_value, na], dtype=data_missing.dtype)
        result = arr.take([-1, 1], fill_value=fill_value, allow_fill=True)
        expected = arr.take([1, 1])
        tm.assert_extension_array_equal(result, expected)

    def test_take_pandas_style_negative_raises(self, data):
        if False:
            for i in range(10):
                print('nop')
        na_value = data.dtype.na_value
        with pytest.raises(ValueError, match=''):
            data.take([0, -2], fill_value=na_value, allow_fill=True)

    @pytest.mark.parametrize('allow_fill', [True, False])
    def test_take_out_of_bounds_raises(self, data, allow_fill):
        if False:
            i = 10
            return i + 15
        arr = data[:3]
        with pytest.raises(IndexError, match='out of bounds|out-of-bounds'):
            arr.take(np.asarray([0, 3]), allow_fill=allow_fill)

    def test_take_series(self, data):
        if False:
            while True:
                i = 10
        s = pd.Series(data)
        result = s.take([0, -1])
        expected = pd.Series(data._from_sequence([data[0], data[len(data) - 1]], dtype=s.dtype), index=[0, len(data) - 1])
        tm.assert_series_equal(result, expected)

    def test_reindex(self, data):
        if False:
            while True:
                i = 10
        na_value = data.dtype.na_value
        s = pd.Series(data)
        result = s.reindex([0, 1, 3])
        expected = pd.Series(data.take([0, 1, 3]), index=[0, 1, 3])
        tm.assert_series_equal(result, expected)
        n = len(data)
        result = s.reindex([-1, 0, n])
        expected = pd.Series(data._from_sequence([na_value, data[0], na_value], dtype=s.dtype), index=[-1, 0, n])
        tm.assert_series_equal(result, expected)
        result = s.reindex([n, n + 1])
        expected = pd.Series(data._from_sequence([na_value, na_value], dtype=s.dtype), index=[n, n + 1])
        tm.assert_series_equal(result, expected)

    def test_reindex_non_na_fill_value(self, data_missing):
        if False:
            for i in range(10):
                print('nop')
        valid = data_missing[1]
        na = data_missing[0]
        arr = data_missing._from_sequence([na, valid], dtype=data_missing.dtype)
        ser = pd.Series(arr)
        result = ser.reindex([0, 1, 2], fill_value=valid)
        expected = pd.Series(data_missing._from_sequence([na, valid, valid], dtype=data_missing.dtype))
        tm.assert_series_equal(result, expected)

    def test_loc_len1(self, data):
        if False:
            for i in range(10):
                print('nop')
        df = pd.DataFrame({'A': data})
        res = df.loc[[0], 'A']
        assert res.ndim == 1
        assert res._mgr.arrays[0].ndim == 1
        if hasattr(res._mgr, 'blocks'):
            assert res._mgr._block.ndim == 1

    def test_item(self, data):
        if False:
            return 10
        s = pd.Series(data)
        result = s[:1].item()
        assert result == data[0]
        msg = 'can only convert an array of size 1 to a Python scalar'
        with pytest.raises(ValueError, match=msg):
            s[:0].item()
        with pytest.raises(ValueError, match=msg):
            s.item()
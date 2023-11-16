from itertools import product
from operator import eq, ne
import warnings
import numpy as np
from toolz import take
from zipline.lib.labelarray import LabelArray
from zipline.testing import check_arrays, parameter_space, ZiplineTestCase
from zipline.testing.predicates import assert_equal
from zipline.utils.compat import unicode

def rotN(l, N):
    if False:
        i = 10
        return i + 15
    "\n    Rotate a list of elements.\n\n    Pulls N elements off the end of the list and appends them to the front.\n\n    >>> rotN(['a', 'b', 'c', 'd'], 2)\n    ['c', 'd', 'a', 'b']\n    >>> rotN(['a', 'b', 'c', 'd'], 3)\n    ['d', 'a', 'b', 'c']\n    "
    assert len(l) >= N, "Can't rotate list by longer than its length."
    return l[N:] + l[:N]

def all_ufuncs():
    if False:
        i = 10
        return i + 15
    ufunc_type = type(np.isnan)
    return (f for f in vars(np).values() if isinstance(f, ufunc_type))

class LabelArrayTestCase(ZiplineTestCase):

    @classmethod
    def init_class_fixtures(cls):
        if False:
            print('Hello World!')
        super(LabelArrayTestCase, cls).init_class_fixtures()
        cls.rowvalues = row = ['', 'a', 'b', 'ab', 'a', '', 'b', 'ab', 'z']
        cls.strs = np.array([rotN(row, i) for i in range(3)], dtype=object)

    def test_fail_on_direct_construction(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError) as e:
            np.ndarray.__new__(LabelArray, (5, 5))
        self.assertEqual(str(e.exception), 'Direct construction of LabelArrays is not supported.')

    @parameter_space(__fail_fast=True, compval=['', 'a', 'z', 'not in the array'], shape=[(27,), (3, 9), (3, 3, 3)], array_astype=(bytes, unicode, object), missing_value=('', 'a', 'not in the array', None))
    def test_compare_to_str(self, compval, shape, array_astype, missing_value):
        if False:
            for i in range(10):
                print('nop')
        strs = self.strs.reshape(shape).astype(array_astype)
        if missing_value is None:
            notmissing = np.not_equal(strs, missing_value)
        else:
            if not isinstance(missing_value, array_astype):
                missing_value = array_astype(missing_value, 'utf-8')
            notmissing = strs != missing_value
        arr = LabelArray(strs, missing_value=missing_value)
        if not isinstance(compval, array_astype):
            compval = array_astype(compval, 'utf-8')
        check_arrays(arr == compval, (strs == compval) & notmissing)
        check_arrays(arr != compval, (strs != compval) & notmissing)
        np_startswith = np.vectorize(lambda elem: elem.startswith(compval))
        check_arrays(arr.startswith(compval), np_startswith(strs) & notmissing)
        np_endswith = np.vectorize(lambda elem: elem.endswith(compval))
        check_arrays(arr.endswith(compval), np_endswith(strs) & notmissing)
        np_contains = np.vectorize(lambda elem: compval in elem)
        check_arrays(arr.has_substring(compval), np_contains(strs) & notmissing)

    @parameter_space(__fail_fast=True, f=[lambda s: str(len(s)), lambda s: s[0], lambda s: ''.join(reversed(s)), lambda s: ''])
    def test_map(self, f):
        if False:
            while True:
                i = 10
        data = np.array([['E', 'GHIJ', 'HIJKLMNOP', 'DEFGHIJ'], ['CDE', 'ABCDEFGHIJKLMNOPQ', 'DEFGHIJKLMNOPQRS', 'ABCDEFGHIJK'], ['DEFGHIJKLMNOPQR', 'DEFGHI', 'DEFGHIJ', 'FGHIJK'], ['EFGHIJKLM', 'EFGHIJKLMNOPQRS', 'ABCDEFGHI', 'DEFGHIJ']], dtype=object)
        la = LabelArray(data, missing_value=None)
        numpy_transformed = np.vectorize(f)(data)
        la_transformed = la.map(f).as_string_array()
        assert_equal(numpy_transformed, la_transformed)

    @parameter_space(missing=['A', None])
    def test_map_ignores_missing_value(self, missing):
        if False:
            for i in range(10):
                print('nop')
        data = np.array([missing, 'B', 'C'], dtype=object)
        la = LabelArray(data, missing_value=missing)

        def increment_char(c):
            if False:
                for i in range(10):
                    print('nop')
            return chr(ord(c) + 1)
        result = la.map(increment_char)
        expected = LabelArray([missing, 'C', 'D'], missing_value=missing)
        assert_equal(result.as_string_array(), expected.as_string_array())

    @parameter_space(__fail_fast=True, f=[lambda s: 0, lambda s: 0.0, lambda s: object()])
    def test_map_requires_f_to_return_a_string_or_none(self, f):
        if False:
            i = 10
            return i + 15
        la = LabelArray(self.strs, missing_value=None)
        with self.assertRaises(TypeError):
            la.map(f)

    def test_map_can_only_return_none_if_missing_value_is_none(self):
        if False:
            print('Hello World!')
        la = LabelArray(self.strs, missing_value=None)
        result = la.map(lambda x: None)
        check_arrays(result, LabelArray(np.full_like(self.strs, None), missing_value=None))
        la = LabelArray(self.strs, missing_value='__MISSING__')
        with self.assertRaises(TypeError):
            la.map(lambda x: None)

    @parameter_space(__fail_fast=True, missing_value=('', 'a', 'not in the array', None))
    def test_compare_to_str_array(self, missing_value):
        if False:
            for i in range(10):
                print('nop')
        strs = self.strs
        shape = strs.shape
        arr = LabelArray(strs, missing_value=missing_value)
        if missing_value is None:
            notmissing = np.not_equal(strs, missing_value)
        else:
            notmissing = strs != missing_value
        check_arrays(arr.not_missing(), notmissing)
        check_arrays(arr.is_missing(), ~notmissing)
        check_arrays(strs == arr, notmissing)
        check_arrays(strs != arr, np.zeros_like(strs, dtype=bool))

        def broadcastable_row(value, dtype):
            if False:
                print('Hello World!')
            return np.full((shape[0], 1), value, dtype=strs.dtype)

        def broadcastable_col(value, dtype):
            if False:
                i = 10
                return i + 15
            return np.full((1, shape[1]), value, dtype=strs.dtype)
        for (comparator, dtype, value) in product((eq, ne), (bytes, unicode, object), set(self.rowvalues)):
            check_arrays(comparator(arr, np.full_like(strs, value)), comparator(strs, value) & notmissing)
            check_arrays(comparator(arr, broadcastable_row(value, dtype=dtype)), comparator(strs, value) & notmissing)
            check_arrays(comparator(arr, broadcastable_col(value, dtype=dtype)), comparator(strs, value) & notmissing)

    @parameter_space(__fail_fast=True, slice_=[0, 1, -1, slice(None), slice(0, 0), slice(0, 3), slice(1, 4), slice(0), slice(None, 1), slice(0, 4, 2), (slice(None), 1), (slice(None), slice(None)), (slice(None), slice(1, 2))])
    def test_slicing_preserves_attributes(self, slice_):
        if False:
            return 10
        arr = LabelArray(self.strs.reshape((9, 3)), missing_value='')
        sliced = arr[slice_]
        self.assertIsInstance(sliced, LabelArray)
        self.assertIs(sliced.categories, arr.categories)
        self.assertIs(sliced.reverse_categories, arr.reverse_categories)
        self.assertIs(sliced.missing_value, arr.missing_value)

    def test_infer_categories(self):
        if False:
            print('Hello World!')
        "\n        Test that categories are inferred in sorted order if they're not\n        explicitly passed.\n        "
        arr1d = LabelArray(self.strs, missing_value='')
        codes1d = arr1d.as_int_array()
        self.assertEqual(arr1d.shape, self.strs.shape)
        self.assertEqual(arr1d.shape, codes1d.shape)
        categories = arr1d.categories
        unique_rowvalues = set(self.rowvalues)
        self.assertEqual(list(categories), sorted(set(self.rowvalues)))
        self.assertEqual(set(codes1d.ravel()), set(range(len(unique_rowvalues))))
        for (idx, value) in enumerate(arr1d.categories):
            check_arrays(self.strs == value, arr1d.as_int_array() == idx)
        arr1d_explicit_categories = LabelArray(self.strs, missing_value='', categories=arr1d.categories)
        check_arrays(arr1d, arr1d_explicit_categories)
        for shape in ((9, 3), (3, 9), (3, 3, 3)):
            strs2d = self.strs.reshape(shape)
            arr2d = LabelArray(strs2d, missing_value='')
            codes2d = arr2d.as_int_array()
            self.assertEqual(arr2d.shape, shape)
            check_arrays(arr2d.categories, categories)
            for (idx, value) in enumerate(arr2d.categories):
                check_arrays(strs2d == value, codes2d == idx)

    def test_reject_ufuncs(self):
        if False:
            i = 10
            return i + 15
        '\n        The internal values of a LabelArray should be opaque to numpy ufuncs.\n\n        Test that all unfuncs fail.\n        '
        labels = LabelArray(self.strs, '')
        ints = np.arange(len(labels))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='unorderable dtypes.*', category=DeprecationWarning)
            warnings.filterwarnings('ignore', message='elementwise comparison failed.*', category=FutureWarning)
            for func in all_ufuncs():
                try:
                    if func.nin == 1:
                        ret = func(labels)
                    elif func.nin == 2:
                        ret = func(labels, ints)
                    else:
                        self.fail('Who added a ternary ufunc !?!')
                except (TypeError, ValueError):
                    pass
                else:
                    self.assertIs(ret, NotImplemented)

    @parameter_space(__fail_fast=True, val=['', 'a', 'not in the array', None], missing_value=['', 'a', 'not in the array', None])
    def test_setitem_scalar(self, val, missing_value):
        if False:
            while True:
                i = 10
        arr = LabelArray(self.strs, missing_value=missing_value)
        if not arr.has_label(val):
            self.assertTrue(val == 'not in the array' or (val is None and missing_value is not None))
            for slicer in [(0, 0), (0, 1), 1]:
                with self.assertRaises(ValueError):
                    arr[slicer] = val
            return
        arr[0, 0] = val
        self.assertEqual(arr[0, 0], val)
        arr[0, 1] = val
        self.assertEqual(arr[0, 1], val)
        arr[1] = val
        if val == missing_value:
            self.assertTrue(arr.is_missing()[1].all())
        else:
            self.assertTrue((arr[1] == val).all())
            self.assertTrue((arr[1].as_string_array() == val).all())
        arr[:, -1] = val
        if val == missing_value:
            self.assertTrue(arr.is_missing()[:, -1].all())
        else:
            self.assertTrue((arr[:, -1] == val).all())
            self.assertTrue((arr[:, -1].as_string_array() == val).all())
        arr[:] = val
        if val == missing_value:
            self.assertTrue(arr.is_missing().all())
        else:
            self.assertFalse(arr.is_missing().any())
            self.assertTrue((arr == val).all())

    def test_setitem_array(self):
        if False:
            while True:
                i = 10
        arr = LabelArray(self.strs, missing_value=None)
        orig_arr = arr.copy()
        self.assertFalse((arr[0] == arr[1]).all(), "This test doesn't test anything because rows 0 and 1 are already equal!")
        arr[0] = arr[1]
        for i in range(arr.shape[1]):
            self.assertEqual(arr[0, i], arr[1, i])
        self.assertFalse((arr[:, 0] == arr[:, 1]).all(), "This test doesn't test anything because columns 0 and 1 are already equal!")
        arr[:, 0] = arr[:, 1]
        for i in range(arr.shape[0]):
            self.assertEqual(arr[i, 0], arr[i, 1])
        arr[:] = orig_arr
        check_arrays(arr, orig_arr)

    @staticmethod
    def check_roundtrip(arr):
        if False:
            i = 10
            return i + 15
        assert_equal(arr.as_string_array(), LabelArray(arr.as_string_array(), arr.missing_value).as_string_array())

    @staticmethod
    def create_categories(width, plus_one):
        if False:
            for i in range(10):
                print('nop')
        length = int(width / 8) + plus_one
        return [''.join(cs) for cs in take(2 ** width + plus_one, product([chr(c) for c in range(256)], repeat=length))]

    def test_narrow_code_storage(self):
        if False:
            while True:
                i = 10
        create_categories = self.create_categories
        check_roundtrip = self.check_roundtrip
        categories = create_categories(8, plus_one=False)
        arr = LabelArray(categories, missing_value=categories[0], categories=categories)
        self.assertEqual(arr.itemsize, 1)
        check_roundtrip(arr)
        arr = LabelArray(categories, missing_value=categories[0])
        self.assertEqual(arr.itemsize, 1)
        check_roundtrip(arr)
        categories = create_categories(8, plus_one=True)
        arr = LabelArray(categories, missing_value=categories[0], categories=categories)
        self.assertEqual(arr.itemsize, 2)
        check_roundtrip(arr)
        categories = create_categories(16, plus_one=False)
        arr = LabelArray(categories, missing_value=categories[0], categories=categories)
        self.assertEqual(arr.itemsize, 2)
        check_roundtrip(arr)
        arr = LabelArray(categories, missing_value=categories[0])
        self.assertEqual(arr.itemsize, 2)
        check_roundtrip(arr)
        categories = create_categories(16, plus_one=True)
        arr = LabelArray(categories, missing_value=categories[0], categories=categories)
        self.assertEqual(arr.itemsize, 4)
        check_roundtrip(arr)
        arr = LabelArray(categories, missing_value=categories[0])
        self.assertEqual(arr.itemsize, 4)
        check_roundtrip(arr)

    def test_known_categories_without_missing_at_boundary(self):
        if False:
            for i in range(10):
                print('nop')
        categories = self.create_categories(8, plus_one=False)
        arr = LabelArray(categories, None, categories=categories)
        self.check_roundtrip(arr)
        self.assertEqual(arr.itemsize, 2)

    def test_narrow_condense_back_to_valid_size(self):
        if False:
            while True:
                i = 10
        categories = ['a'] * (2 ** 8 + 1)
        arr = LabelArray(categories, missing_value=categories[0])
        assert_equal(arr.itemsize, 1)
        self.check_roundtrip(arr)
        categories = self.create_categories(16, plus_one=False)
        categories.append(categories[0])
        arr = LabelArray(categories, missing_value=categories[0])
        assert_equal(arr.itemsize, 2)
        self.check_roundtrip(arr)

    def test_map_shrinks_code_storage_if_possible(self):
        if False:
            print('Hello World!')
        arr = LabelArray(self.create_categories(16, plus_one=False)[:-1], missing_value=None)
        self.assertEqual(arr.itemsize, 2)

        def either_A_or_B(s):
            if False:
                i = 10
                return i + 15
            return ('A', 'B')[sum((ord(c) for c in s)) % 2]
        result = arr.map(either_A_or_B)
        self.assertEqual(set(result.categories), {'A', 'B', None})
        self.assertEqual(result.itemsize, 1)
        assert_equal(np.vectorize(either_A_or_B)(arr.as_string_array()), result.as_string_array())

    def test_map_never_increases_code_storage_size(self):
        if False:
            for i in range(10):
                print('nop')
        categories = self.create_categories(8, plus_one=False)[:-1]
        larger_categories = self.create_categories(16, plus_one=False)
        categories_twice = categories + categories
        arr = LabelArray(categories_twice, missing_value=None)
        assert_equal(arr.itemsize, 1)
        gen_unique_categories = iter(larger_categories)

        def new_string_every_time(c):
            if False:
                for i in range(10):
                    print('nop')
            return next(gen_unique_categories)
        result = arr.map(new_string_every_time)
        assert_equal(result.itemsize, 1)
        expected = LabelArray(larger_categories[:len(categories)] * 2, missing_value=None)
        assert_equal(result.as_string_array(), expected.as_string_array())

    def manual_narrow_condense_back_to_valid_size_slow(self):
        if False:
            print('Hello World!')
        "This test is really slow so we don't want it run by default.\n        "
        categories = self.create_categories(24, plus_one=False)
        categories.append(categories[0])
        arr = LabelArray(categories, missing_value=categories[0])
        assert_equal(arr.itemsize, 4)
        self.check_roundtrip(arr)

    def test_copy_categories_list(self):
        if False:
            while True:
                i = 10
        'regression test for #1927\n        '
        categories = ['a', 'b', 'c']
        LabelArray([None, 'a', 'b', 'c'], missing_value=None, categories=categories)
        assert_equal(categories, ['a', 'b', 'c'])

    def test_fortran_contiguous_input(self):
        if False:
            print('Hello World!')
        strs = np.array([['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd']], dtype=object)
        strs_F = strs.T
        self.assertTrue(strs_F.flags.f_contiguous)
        arr = LabelArray(strs_F, missing_value=None, categories=['a', 'b', 'c', 'd', None])
        assert_equal(arr.as_string_array(), strs_F)
        arr = LabelArray(strs_F, missing_value=None)
        assert_equal(arr.as_string_array(), strs_F)
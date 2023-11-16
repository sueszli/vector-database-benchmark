import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
X = np.arange(10)

def global_ndarray_func(x):
    if False:
        print('Hello World!')
    y = x + X.shape[0]
    return y
cplx_X = np.arange(10, dtype=np.complex128)
tmp = np.arange(10, dtype=np.complex128)
cplx_X += (tmp + 10) * 1j

def global_cplx_arr_copy(a):
    if False:
        for i in range(10):
            print('nop')
    for i in range(len(a)):
        a[i] = cplx_X[i]
x_dt = np.dtype([('a', np.int32), ('b', np.float32)])
rec_X = np.recarray(10, dtype=x_dt)
for i in range(len(rec_X)):
    rec_X[i].a = i
    rec_X[i].b = i + 0.5

def global_rec_arr_copy(a):
    if False:
        i = 10
        return i + 15
    for i in range(len(a)):
        a[i] = rec_X[i]

def global_rec_arr_extract_fields(a, b):
    if False:
        while True:
            i = 10
    for i in range(len(a)):
        a[i] = rec_X[i].a
        b[i] = rec_X[i].b
y_dt = np.dtype([('c', np.int16), ('d', np.float64)])
rec_Y = np.recarray(10, dtype=y_dt)
for i in range(len(rec_Y)):
    rec_Y[i].c = i + 10
    rec_Y[i].d = i + 10.5

def global_two_rec_arrs(a, b, c, d):
    if False:
        while True:
            i = 10
    for i in range(len(a)):
        a[i] = rec_X[i].a
        b[i] = rec_X[i].b
        c[i] = rec_Y[i].c
        d[i] = rec_Y[i].d
record_only_X = np.recarray(1, dtype=x_dt)[0]
record_only_X.a = 1
record_only_X.b = 1.5

@jit(nopython=True)
def global_record_func(x):
    if False:
        for i in range(10):
            print('nop')
    return x.a == record_only_X.a

@jit(nopython=True)
def global_module_func(x, y):
    if False:
        return 10
    return usecases.andornopython(x, y)
tup_int = (1, 2)
tup_str = ('a', 'b')
tup_mixed = (1, 'a')
tup_float = (1.2, 3.5)
tup_npy_ints = (np.uint64(12), np.int8(3))
tup_tup_array = ((np.ones(5),),)
mixed_tup_tup_array = (('Z', np.ones(5)), 2j, 'A')

def global_int_tuple():
    if False:
        return 10
    return tup_int[0] + tup_int[1]

def global_str_tuple():
    if False:
        for i in range(10):
            print('nop')
    return tup_str[0] + tup_str[1]

def global_mixed_tuple():
    if False:
        print('Hello World!')
    idx = tup_mixed[0]
    field = tup_mixed[1]
    return rec_X[idx][field]

def global_float_tuple():
    if False:
        return 10
    return tup_float[0] + tup_float[1]

def global_npy_int_tuple():
    if False:
        return 10
    return tup_npy_ints[0] + tup_npy_ints[1]

def global_write_to_arr_in_tuple():
    if False:
        return 10
    tup_tup_array[0][0][0] = 10.0

def global_write_to_arr_in_mixed_tuple():
    if False:
        print('Hello World!')
    mixed_tup_tup_array[0][1][0] = 10.0
_glbl_np_bool_T = np.bool_(True)
_glbl_np_bool_F = np.bool_(False)

@register_jitable
def _sink(*args):
    if False:
        i = 10
        return i + 15
    pass

def global_npy_bool():
    if False:
        while True:
            i = 10
    _sink(_glbl_np_bool_T, _glbl_np_bool_F)
    return (_glbl_np_bool_T, _glbl_np_bool_F)

class TestGlobals(unittest.TestCase):

    def check_global_ndarray(self, **jitargs):
        if False:
            print('Hello World!')
        ctestfunc = jit(**jitargs)(global_ndarray_func)
        self.assertEqual(ctestfunc(1), 11)

    def test_global_ndarray(self):
        if False:
            i = 10
            return i + 15
        self.check_global_ndarray(forceobj=True)

    def test_global_ndarray_npm(self):
        if False:
            i = 10
            return i + 15
        self.check_global_ndarray(nopython=True)

    def check_global_complex_arr(self, **jitargs):
        if False:
            i = 10
            return i + 15
        ctestfunc = jit(**jitargs)(global_cplx_arr_copy)
        arr = np.zeros(len(cplx_X), dtype=np.complex128)
        ctestfunc(arr)
        np.testing.assert_equal(arr, cplx_X)

    def test_global_complex_arr(self):
        if False:
            print('Hello World!')
        self.check_global_complex_arr(forceobj=True)

    def test_global_complex_arr_npm(self):
        if False:
            print('Hello World!')
        self.check_global_complex_arr(nopython=True)

    def check_global_rec_arr(self, **jitargs):
        if False:
            print('Hello World!')
        ctestfunc = jit(**jitargs)(global_rec_arr_copy)
        arr = np.zeros(rec_X.shape, dtype=x_dt)
        ctestfunc(arr)
        np.testing.assert_equal(arr, rec_X)

    def test_global_rec_arr(self):
        if False:
            while True:
                i = 10
        self.check_global_rec_arr(forceobj=True)

    def test_global_rec_arr_npm(self):
        if False:
            i = 10
            return i + 15
        self.check_global_rec_arr(nopython=True)

    def check_global_rec_arr_extract(self, **jitargs):
        if False:
            return 10
        ctestfunc = jit(**jitargs)(global_rec_arr_extract_fields)
        arr1 = np.zeros(rec_X.shape, dtype=np.int32)
        arr2 = np.zeros(rec_X.shape, dtype=np.float32)
        ctestfunc(arr1, arr2)
        np.testing.assert_equal(arr1, rec_X.a)
        np.testing.assert_equal(arr2, rec_X.b)

    def test_global_rec_arr_extract(self):
        if False:
            i = 10
            return i + 15
        self.check_global_rec_arr_extract(forceobj=True)

    def test_global_rec_arr_extract_npm(self):
        if False:
            i = 10
            return i + 15
        self.check_global_rec_arr_extract(nopython=True)

    def check_two_global_rec_arrs(self, **jitargs):
        if False:
            while True:
                i = 10
        ctestfunc = jit(**jitargs)(global_two_rec_arrs)
        arr1 = np.zeros(rec_X.shape, dtype=np.int32)
        arr2 = np.zeros(rec_X.shape, dtype=np.float32)
        arr3 = np.zeros(rec_Y.shape, dtype=np.int16)
        arr4 = np.zeros(rec_Y.shape, dtype=np.float64)
        ctestfunc(arr1, arr2, arr3, arr4)
        np.testing.assert_equal(arr1, rec_X.a)
        np.testing.assert_equal(arr2, rec_X.b)
        np.testing.assert_equal(arr3, rec_Y.c)
        np.testing.assert_equal(arr4, rec_Y.d)

    def test_two_global_rec_arrs(self):
        if False:
            while True:
                i = 10
        self.check_two_global_rec_arrs(forceobj=True)

    def test_two_global_rec_arrs_npm(self):
        if False:
            while True:
                i = 10
        self.check_two_global_rec_arrs(nopython=True)

    def test_global_module(self):
        if False:
            print('Hello World!')
        res = global_module_func(5, 6)
        self.assertEqual(True, res)

    def test_global_record(self):
        if False:
            print('Hello World!')
        x = np.recarray(1, dtype=x_dt)[0]
        x.a = 1
        res = global_record_func(x)
        self.assertEqual(True, res)
        x.a = 2
        res = global_record_func(x)
        self.assertEqual(False, res)

    def test_global_int_tuple(self):
        if False:
            i = 10
            return i + 15
        pyfunc = global_int_tuple
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())

    def test_global_str_tuple(self):
        if False:
            i = 10
            return i + 15
        pyfunc = global_str_tuple
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())

    def test_global_mixed_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = global_mixed_tuple
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())

    def test_global_float_tuple(self):
        if False:
            i = 10
            return i + 15
        pyfunc = global_float_tuple
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())

    def test_global_npy_int_tuple(self):
        if False:
            i = 10
            return i + 15
        pyfunc = global_npy_int_tuple
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())

    def test_global_write_to_arr_in_tuple(self):
        if False:
            print('Hello World!')
        for func in (global_write_to_arr_in_tuple, global_write_to_arr_in_mixed_tuple):
            jitfunc = njit(func)
            with self.assertRaises(errors.TypingError) as e:
                jitfunc()
            msg = 'Cannot modify readonly array of type:'
            self.assertIn(msg, str(e.exception))

    def test_global_npy_bool(self):
        if False:
            while True:
                i = 10
        pyfunc = global_npy_bool
        jitfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), jitfunc())
if __name__ == '__main__':
    unittest.main()
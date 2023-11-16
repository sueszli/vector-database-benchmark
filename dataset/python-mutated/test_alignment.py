import numpy as np
from numba import from_dtype, njit, void
from numba.tests.support import TestCase

class TestAlignment(TestCase):

    def test_record_alignment(self):
        if False:
            return 10
        rec_dtype = np.dtype([('a', 'int32'), ('b', 'float64')], align=True)
        rec = from_dtype(rec_dtype)

        @njit((rec[:],))
        def foo(a):
            if False:
                i = 10
                return i + 15
            for i in range(a.size):
                a[i].a = a[i].b
        a_recarray = np.recarray(3, dtype=rec_dtype)
        for i in range(a_recarray.size):
            a_rec = a_recarray[i]
            a_rec.a = 0
            a_rec.b = (i + 1) * 123
        foo(a_recarray)
        np.testing.assert_equal(a_recarray.a, a_recarray.b)

    def test_record_misaligned(self):
        if False:
            for i in range(10):
                print('nop')
        rec_dtype = np.dtype([('a', 'int32'), ('b', 'float64')])
        rec = from_dtype(rec_dtype)

        @njit((rec[:],))
        def foo(a):
            if False:
                for i in range(10):
                    print('nop')
            for i in range(a.size):
                a[i].a = a[i].b
if __name__ == '__main__':
    unittest.main()
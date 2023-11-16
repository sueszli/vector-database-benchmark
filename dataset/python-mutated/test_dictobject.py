"""
Testing numba implementation of the numba dictionary.

The tests here only check that the numba typing and codegen are working
correctly.  Detailed testing of the underlying dictionary operations is done
in test_dictimpl.py.
"""
import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin, unittest, override_config, forbid_codegen
from numba.experimental import jitclass
from numba.extending import overload

class TestDictObject(MemoryLeakMixin, TestCase):

    def test_dict_bool(self):
        if False:
            return 10
        '\n        Exercise bool(dict)\n        '

        @njit
        def foo(n):
            if False:
                return 10
            d = dictobject.new_dict(int32, float32)
            for i in range(n):
                d[i] = i + 1
            return bool(d)
        self.assertEqual(foo(n=0), False)
        self.assertEqual(foo(n=1), True)
        self.assertEqual(foo(n=2), True)
        self.assertEqual(foo(n=100), True)

    def test_dict_create(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exercise dictionary creation, insertion and len\n        '

        @njit
        def foo(n):
            if False:
                return 10
            d = dictobject.new_dict(int32, float32)
            for i in range(n):
                d[i] = i + 1
            return len(d)
        self.assertEqual(foo(n=0), 0)
        self.assertEqual(foo(n=1), 1)
        self.assertEqual(foo(n=2), 2)
        self.assertEqual(foo(n=100), 100)

    def test_dict_get(self):
        if False:
            while True:
                i = 10
        '\n        Exercise dictionary creation, insertion and get\n        '

        @njit
        def foo(n, targets):
            if False:
                i = 10
                return i + 15
            d = dictobject.new_dict(int32, float64)
            for i in range(n):
                d[i] = i
            output = []
            for t in targets:
                output.append(d.get(t))
            return output
        self.assertEqual(foo(5, [0, 1, 9]), [0, 1, None])
        self.assertEqual(foo(10, [0, 1, 9]), [0, 1, 9])
        self.assertEqual(foo(10, [-1, 9, 1]), [None, 9, 1])

    def test_dict_get_with_default(self):
        if False:
            while True:
                i = 10
        '\n        Exercise dict.get(k, d) where d is set\n        '

        @njit
        def foo(n, target, default):
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            for i in range(n):
                d[i] = i
            return d.get(target, default)
        self.assertEqual(foo(5, 3, -1), 3)
        self.assertEqual(foo(5, 5, -1), -1)

    def test_dict_getitem(self):
        if False:
            return 10
        '\n        Exercise dictionary __getitem__\n        '

        @njit
        def foo(keys, vals, target):
            if False:
                while True:
                    i = 10
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            return d[target]
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals, 1), 0.1)
        self.assertEqual(foo(keys, vals, 2), 0.2)
        self.assertEqual(foo(keys, vals, 3), 0.3)
        self.assert_no_memory_leak()
        self.disable_leak_check()
        with self.assertRaises(KeyError):
            foo(keys, vals, 0)
        with self.assertRaises(KeyError):
            foo(keys, vals, 4)

    def test_dict_popitem(self):
        if False:
            return 10
        '\n        Exercise dictionary .popitem\n        '

        @njit
        def foo(keys, vals):
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            return d.popitem()
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        for i in range(1, len(keys)):
            self.assertEqual(foo(keys[:i], vals[:i]), (keys[i - 1], vals[i - 1]))

    def test_dict_popitem_many(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exercise dictionary .popitem\n        '

        @njit
        def core(d, npop):
            if False:
                while True:
                    i = 10
            (keysum, valsum) = (0, 0)
            for _ in range(npop):
                (k, v) = d.popitem()
                keysum += k
                valsum -= v
            return (keysum, valsum)

        @njit
        def foo(keys, vals, npop):
            if False:
                for i in range(10):
                    print('nop')
            d = dictobject.new_dict(int32, int32)
            for (k, v) in zip(keys, vals):
                d[k] = v
            return core(d, npop)
        keys = [1, 2, 3]
        vals = [10, 20, 30]
        for i in range(len(keys)):
            self.assertEqual(foo(keys, vals, npop=3), core.py_func(dict(zip(keys, vals)), npop=3))
        self.assert_no_memory_leak()
        self.disable_leak_check()
        with self.assertRaises(KeyError):
            foo(keys, vals, npop=4)

    def test_dict_pop(self):
        if False:
            print('Hello World!')
        '\n        Exercise dictionary .pop\n        '

        @njit
        def foo(keys, vals, target):
            if False:
                i = 10
                return i + 15
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            return (d.pop(target, None), len(d))
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals, 1), (0.1, 2))
        self.assertEqual(foo(keys, vals, 2), (0.2, 2))
        self.assertEqual(foo(keys, vals, 3), (0.3, 2))
        self.assertEqual(foo(keys, vals, 0), (None, 3))
        self.assert_no_memory_leak()
        self.disable_leak_check()

        @njit
        def foo():
            if False:
                return 10
            d = dictobject.new_dict(int32, float64)
            return d.pop(0)
        with self.assertRaises(KeyError):
            foo()

    def test_dict_pop_many(self):
        if False:
            i = 10
            return i + 15
        '\n        Exercise dictionary .pop\n        '

        @njit
        def core(d, pops):
            if False:
                while True:
                    i = 10
            total = 0
            for k in pops:
                total += k + d.pop(k, 0.123) + len(d)
                total *= 2
            return total

        @njit
        def foo(keys, vals, pops):
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            return core(d, pops)
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        pops = [2, 3, 3, 1, 0, 2, 1, 0, -1]
        self.assertEqual(foo(keys, vals, pops), core.py_func(dict(zip(keys, vals)), pops))

    def test_dict_delitem(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(keys, vals, target):
            if False:
                return 10
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            del d[target]
            return (len(d), d.get(target))
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals, 1), (2, None))
        self.assertEqual(foo(keys, vals, 2), (2, None))
        self.assertEqual(foo(keys, vals, 3), (2, None))
        self.assert_no_memory_leak()
        self.disable_leak_check()
        with self.assertRaises(KeyError):
            foo(keys, vals, 0)

    def test_dict_clear(self):
        if False:
            while True:
                i = 10
        '\n        Exercise dict.clear\n        '

        @njit
        def foo(keys, vals):
            if False:
                for i in range(10):
                    print('nop')
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            b4 = len(d)
            d.clear()
            return (b4, len(d))
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals), (3, 0))

    def test_dict_items(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exercise dict.items\n        '

        @njit
        def foo(keys, vals):
            if False:
                while True:
                    i = 10
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            out = []
            for kv in d.items():
                out.append(kv)
            return out
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals), list(zip(keys, vals)))

        @njit
        def foo():
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            out = []
            for kv in d.items():
                out.append(kv)
            return out
        self.assertEqual(foo(), [])

    def test_dict_keys(self):
        if False:
            print('Hello World!')
        '\n        Exercise dict.keys\n        '

        @njit
        def foo(keys, vals):
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            out = []
            for k in d.keys():
                out.append(k)
            return out
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals), keys)

    def test_dict_keys_len(self):
        if False:
            print('Hello World!')
        '\n        Exercise len(dict.keys())\n        '

        @njit
        def foo(keys, vals):
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            return len(d.keys())
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals), len(keys))

    def test_dict_values(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exercise dict.values\n        '

        @njit
        def foo(keys, vals):
            if False:
                i = 10
                return i + 15
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            out = []
            for v in d.values():
                out.append(v)
            return out
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals), vals)

    def test_dict_values_len(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exercise len(dict.values())\n        '

        @njit
        def foo(keys, vals):
            if False:
                for i in range(10):
                    print('nop')
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            return len(d.values())
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals), len(vals))

    def test_dict_items_len(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exercise len(dict.items())\n        '

        @njit
        def foo(keys, vals):
            if False:
                while True:
                    i = 10
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            return len(d.items())
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertPreciseEqual(foo(keys, vals), len(vals))

    def test_dict_iter(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exercise iter(dict)\n        '

        @njit
        def foo(keys, vals):
            if False:
                while True:
                    i = 10
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            out = []
            for k in d:
                out.append(k)
            return out
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals), [1, 2, 3])

    def test_dict_contains(self):
        if False:
            return 10
        '\n        Exercise operator.contains\n        '

        @njit
        def foo(keys, vals, checklist):
            if False:
                return 10
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            out = []
            for k in checklist:
                out.append(k in d)
            return out
        keys = [1, 2, 3]
        vals = [0.1, 0.2, 0.3]
        self.assertEqual(foo(keys, vals, [2, 3, 4, 1, 0]), [True, True, False, True, False])

    def test_dict_copy(self):
        if False:
            return 10
        '\n        Exercise dict.copy\n        '

        @njit
        def foo(keys, vals):
            if False:
                while True:
                    i = 10
            d = dictobject.new_dict(int32, float64)
            for (k, v) in zip(keys, vals):
                d[k] = v
            return list(d.copy().items())
        keys = list(range(20))
        vals = [x + i / 100 for (i, x) in enumerate(keys)]
        out = foo(keys, vals)
        self.assertEqual(out, list(zip(keys, vals)))

    def test_dict_setdefault(self):
        if False:
            while True:
                i = 10
        '\n        Exercise dict.setdefault\n        '

        @njit
        def foo():
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            d.setdefault(1, 1.2)
            a = d.get(1)
            d[1] = 2.3
            b = d.get(1)
            d[2] = 3.4
            d.setdefault(2, 4.5)
            c = d.get(2)
            return (a, b, c)
        self.assertEqual(foo(), (1.2, 2.3, 3.4))

    def test_dict_equality(self):
        if False:
            while True:
                i = 10
        '\n        Exercise dict.__eq__ and .__ne__\n        '

        @njit
        def foo(na, nb, fa, fb):
            if False:
                print('Hello World!')
            da = dictobject.new_dict(int32, float64)
            db = dictobject.new_dict(int32, float64)
            for i in range(na):
                da[i] = i * fa
            for i in range(nb):
                db[i] = i * fb
            return (da == db, da != db)
        self.assertEqual(foo(10, 10, 3, 3), (True, False))
        self.assertEqual(foo(10, 10, 3, 3.1), (False, True))
        self.assertEqual(foo(11, 10, 3, 3), (False, True))
        self.assertEqual(foo(10, 11, 3, 3), (False, True))

    def test_dict_equality_more(self):
        if False:
            return 10
        '\n        Exercise dict.__eq__\n        '

        @njit
        def foo(ak, av, bk, bv):
            if False:
                return 10
            da = dictobject.new_dict(int32, float64)
            db = dictobject.new_dict(int64, float32)
            for i in range(len(ak)):
                da[ak[i]] = av[i]
            for i in range(len(bk)):
                db[bk[i]] = bv[i]
            return da == db
        ak = [1, 2, 3]
        av = [2, 3, 4]
        bk = [1, 2, 3]
        bv = [2, 3, 4]
        self.assertTrue(foo(ak, av, bk, bv))
        ak = [1, 2, 3]
        av = [2, 3, 4]
        bk = [1, 2, 2, 3]
        bv = [2, 1, 3, 4]
        self.assertTrue(foo(ak, av, bk, bv))
        ak = [1, 2, 3]
        av = [2, 3, 4]
        bk = [1, 2, 3]
        bv = [2, 1, 4]
        self.assertFalse(foo(ak, av, bk, bv))
        ak = [0, 2, 3]
        av = [2, 3, 4]
        bk = [1, 2, 3]
        bv = [2, 3, 4]
        self.assertFalse(foo(ak, av, bk, bv))

    def test_dict_equality_diff_type(self):
        if False:
            return 10
        '\n        Exercise dict.__eq__\n        '

        @njit
        def foo(na, b):
            if False:
                for i in range(10):
                    print('nop')
            da = dictobject.new_dict(int32, float64)
            for i in range(na):
                da[i] = i
            return da == b
        self.assertFalse(foo(10, 1))
        self.assertFalse(foo(10, (1,)))

    def test_dict_to_from_meminfo(self):
        if False:
            i = 10
            return i + 15
        '\n        Exercise dictobject.{_as_meminfo, _from_meminfo}\n        '

        @njit
        def make_content(nelem):
            if False:
                print('Hello World!')
            for i in range(nelem):
                yield (i, i + (i + 1) / 100)

        @njit
        def boxer(nelem):
            if False:
                return 10
            d = dictobject.new_dict(int32, float64)
            for (k, v) in make_content(nelem):
                d[k] = v
            return dictobject._as_meminfo(d)
        dcttype = types.DictType(int32, float64)

        @njit
        def unboxer(mi):
            if False:
                i = 10
                return i + 15
            d = dictobject._from_meminfo(mi, dcttype)
            return list(d.items())
        mi = boxer(10)
        self.assertEqual(mi.refcount, 1)
        got = unboxer(mi)
        expected = list(make_content.py_func(10))
        self.assertEqual(got, expected)

    def test_001_cannot_downcast_key(self):
        if False:
            return 10

        @njit
        def foo(n):
            if False:
                for i in range(10):
                    print('nop')
            d = dictobject.new_dict(int32, float64)
            for i in range(n):
                d[i] = i + 1
            z = d.get(1j)
            return z
        with self.assertRaises(TypingError) as raises:
            foo(10)
        self.assertIn('cannot safely cast complex128 to int32', str(raises.exception))

    def test_002_cannot_downcast_default(self):
        if False:
            while True:
                i = 10

        @njit
        def foo(n):
            if False:
                i = 10
                return i + 15
            d = dictobject.new_dict(int32, float64)
            for i in range(n):
                d[i] = i + 1
            z = d.get(2 * n, 1j)
            return z
        with self.assertRaises(TypingError) as raises:
            foo(10)
        self.assertIn('cannot safely cast complex128 to float64', str(raises.exception))

    def test_003_cannot_downcast_key(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(n):
            if False:
                while True:
                    i = 10
            d = dictobject.new_dict(int32, float64)
            for i in range(n):
                d[i] = i + 1
            z = d.get(2.4)
            return z
        with self.assertRaises(TypingError) as raises:
            foo(10)
        self.assertIn('cannot safely cast float64 to int32', str(raises.exception))

    def test_004_cannot_downcast_key(self):
        if False:
            print('Hello World!')

        @njit
        def foo():
            if False:
                return 10
            d = dictobject.new_dict(int32, float64)
            d[1j] = 7.0
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('cannot safely cast complex128 to int32', str(raises.exception))

    def test_005_cannot_downcast_value(self):
        if False:
            print('Hello World!')

        @njit
        def foo():
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            d[1] = 1j
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('cannot safely cast complex128 to float64', str(raises.exception))

    def test_006_cannot_downcast_key(self):
        if False:
            print('Hello World!')

        @njit
        def foo():
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            d[11.5]
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('cannot safely cast float64 to int32', str(raises.exception))

    @unittest.skipUnless(sys.maxsize > 2 ** 32, '64 bit test only')
    def test_007_collision_checks(self):
        if False:
            return 10

        @njit
        def foo(v1, v2):
            if False:
                return 10
            d = dictobject.new_dict(int64, float64)
            c1 = np.uint64(2 ** 61 - 1)
            c2 = np.uint64(0)
            assert hash(c1) == hash(c2)
            d[c1] = v1
            d[c2] = v2
            return (d[c1], d[c2])
        (a, b) = (10.0, 20.0)
        (x, y) = foo(a, b)
        self.assertEqual(x, a)
        self.assertEqual(y, b)

    def test_008_lifo_popitem(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(n):
            if False:
                i = 10
                return i + 15
            d = dictobject.new_dict(int32, float64)
            for i in range(n):
                d[i] = i + 1
            keys = []
            vals = []
            for i in range(n):
                tmp = d.popitem()
                keys.append(tmp[0])
                vals.append(tmp[1])
            return (keys, vals)
        z = 10
        (gk, gv) = foo(z)
        self.assertEqual(gk, [x for x in reversed(range(z))])
        self.assertEqual(gv, [x + 1 for x in reversed(range(z))])

    def test_010_cannot_downcast_default(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            d[0] = 6.0
            d[1] = 7.0
            d.pop(11, 12j)
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('cannot safely cast complex128 to float64', str(raises.exception))

    def test_011_cannot_downcast_key(self):
        if False:
            print('Hello World!')

        @njit
        def foo():
            if False:
                return 10
            d = dictobject.new_dict(int32, float64)
            d[0] = 6.0
            d[1] = 7.0
            d.pop(11j)
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('cannot safely cast complex128 to int32', str(raises.exception))

    def test_012_cannot_downcast_key(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo():
            if False:
                while True:
                    i = 10
            d = dictobject.new_dict(int32, float64)
            d[0] = 6.0
            return 1j in d
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('cannot safely cast complex128 to int32', str(raises.exception))

    def test_013_contains_empty_dict(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            d = dictobject.new_dict(int32, float64)
            return 1 in d
        self.assertFalse(foo())

    def test_014_not_contains_empty_dict(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                return 10
            d = dictobject.new_dict(int32, float64)
            return 1 not in d
        self.assertTrue(foo())

    def test_015_dict_clear(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo(n):
            if False:
                return 10
            d = dictobject.new_dict(int32, float64)
            for i in range(n):
                d[i] = i + 1
            x = len(d)
            d.clear()
            y = len(d)
            return (x, y)
        m = 10
        self.assertEqual(foo(m), (m, 0))

    def test_016_cannot_downcast_key(self):
        if False:
            return 10

        @njit
        def foo():
            if False:
                return 10
            d = dictobject.new_dict(int32, float64)
            d.setdefault(1j, 12.0)
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('cannot safely cast complex128 to int32', str(raises.exception))

    def test_017_cannot_downcast_default(self):
        if False:
            return 10

        @njit
        def foo():
            if False:
                while True:
                    i = 10
            d = dictobject.new_dict(int32, float64)
            d.setdefault(1, 12j)
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('cannot safely cast complex128 to float64', str(raises.exception))

    def test_018_keys_iter_are_views(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            d[11] = 12.0
            k1 = d.keys()
            d[22] = 9.0
            k2 = d.keys()
            rk1 = [x for x in k1]
            rk2 = [x for x in k2]
            return (rk1, rk2)
        (a, b) = foo()
        self.assertEqual(a, b)
        self.assertEqual(a, [11, 22])

    @unittest.expectedFailure
    def test_019(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            d[11] = 12.0
            d[22] = 9.0
            k2 = d.keys() & {12}
            return k2
        print(foo())

    def test_020_string_key(self):
        if False:
            while True:
                i = 10

        @njit
        def foo():
            if False:
                i = 10
                return i + 15
            d = dictobject.new_dict(types.unicode_type, float64)
            d['a'] = 1.0
            d['b'] = 2.0
            d['c'] = 3.0
            d['d'] = 4.0
            out = []
            for x in d.items():
                out.append(x)
            return (out, d['a'])
        (items, da) = foo()
        self.assertEqual(items, [('a', 1.0), ('b', 2.0), ('c', 3.0), ('d', 4)])
        self.assertEqual(da, 1.0)

    def test_021_long_str_key(self):
        if False:
            return 10

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            d = dictobject.new_dict(types.unicode_type, float64)
            tmp = []
            for i in range(10000):
                tmp.append('a')
            s = ''.join(tmp)
            d[s] = 1.0
            out = list(d.items())
            return out
        self.assertEqual(foo(), [('a' * 10000, 1)])

    def test_022_references_juggle(self):
        if False:
            return 10

        @njit
        def foo():
            if False:
                print('Hello World!')
            d = dictobject.new_dict(int32, float64)
            e = d
            d[1] = 12.0
            e[2] = 14.0
            e = dictobject.new_dict(int32, float64)
            e[1] = 100.0
            e[2] = 1000.0
            f = d
            d = e
            k1 = [x for x in d.items()]
            k2 = [x for x in e.items()]
            k3 = [x for x in f.items()]
            return (k1, k2, k3)
        (k1, k2, k3) = foo()
        self.assertEqual(k1, [(1, 100.0), (2, 1000.0)])
        self.assertEqual(k2, [(1, 100.0), (2, 1000.0)])
        self.assertEqual(k3, [(1, 12), (2, 14)])

    def test_023_closure(self):
        if False:
            print('Hello World!')

        @njit
        def foo():
            if False:
                while True:
                    i = 10
            d = dictobject.new_dict(int32, float64)

            def bar():
                if False:
                    return 10
                d[1] = 12.0
                d[2] = 14.0
            bar()
            return [x for x in d.keys()]
        self.assertEqual(foo(), [1, 2])

    def test_024_unicode_getitem_keys(self):
        if False:
            while True:
                i = 10

        @njit
        def foo():
            if False:
                return 10
            s = 'aሴ'
            d = {s[0]: 1}
            return d['a']
        self.assertEqual(foo(), foo.py_func())

        @njit
        def foo():
            if False:
                i = 10
                return i + 15
            s = 'abcሴ'
            d = {s[:1]: 1}
            return d['a']
        self.assertEqual(foo(), foo.py_func())

    def test_issue6570_alignment_padding(self):
        if False:
            print('Hello World!')
        keyty = types.Tuple([types.uint64, types.float32])

        @njit
        def foo():
            if False:
                while True:
                    i = 10
            d = dictobject.new_dict(keyty, float64)
            t1 = np.array([3], dtype=np.uint64)
            t2 = np.array([5.67], dtype=np.float32)
            v1 = np.array([10.23], dtype=np.float32)
            d[t1[0], t2[0]] = v1[0]
            return (t1[0], t2[0]) in d
        self.assertTrue(foo())

    def test_dict_update(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests dict.update works with various dictionaries.\n        '
        n = 10

        def f1(n):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test update with a regular dictionary.\n            '
            d1 = {i: i + 1 for i in range(n)}
            d2 = {3 * i: i for i in range(n)}
            d1.update(d2)
            return d1
        py_func = f1
        cfunc = njit()(f1)
        a = py_func(n)
        b = cfunc(n)
        self.assertEqual(a, b)

        def f2(n):
            if False:
                while True:
                    i = 10
            '\n            Test update where one of the dictionaries\n            is created as a Python literal.\n            '
            d1 = {1: 2, 3: 4, 5: 6}
            d2 = {3 * i: i for i in range(n)}
            d1.update(d2)
            return d1
        py_func = f2
        cfunc = njit()(f2)
        a = py_func(n)
        b = cfunc(n)
        self.assertEqual(a, b)

class TestDictTypeCasting(TestCase):

    def check_good(self, fromty, toty):
        if False:
            for i in range(10):
                print('nop')
        _sentry_safe_cast(fromty, toty)

    def check_bad(self, fromty, toty):
        if False:
            print('Hello World!')
        with self.assertRaises(TypingError) as raises:
            _sentry_safe_cast(fromty, toty)
        self.assertIn('cannot safely cast {fromty} to {toty}'.format(**locals()), str(raises.exception))

    def test_cast_int_to(self):
        if False:
            return 10
        self.check_good(types.int32, types.float32)
        self.check_good(types.int32, types.float64)
        self.check_good(types.int32, types.complex128)
        self.check_good(types.int64, types.complex128)
        self.check_bad(types.int32, types.complex64)
        self.check_good(types.int8, types.complex64)

    def test_cast_float_to(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_good(types.float32, types.float64)
        self.check_good(types.float32, types.complex64)
        self.check_good(types.float64, types.complex128)

    def test_cast_bool_to(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_good(types.boolean, types.int32)
        self.check_good(types.boolean, types.float64)
        self.check_good(types.boolean, types.complex128)

class TestTypedDict(MemoryLeakMixin, TestCase):

    def test_basic(self):
        if False:
            return 10
        d = Dict.empty(int32, float32)
        self.assertEqual(len(d), 0)
        d[1] = 1
        d[2] = 2.3
        d[3] = 3.4
        self.assertEqual(len(d), 3)
        self.assertEqual(list(d.keys()), [1, 2, 3])
        for (x, y) in zip(list(d.values()), [1, 2.3, 3.4]):
            self.assertAlmostEqual(x, y, places=4)
        self.assertAlmostEqual(d[1], 1)
        self.assertAlmostEqual(d[2], 2.3, places=4)
        self.assertAlmostEqual(d[3], 3.4, places=4)
        del d[2]
        self.assertEqual(len(d), 2)
        self.assertIsNone(d.get(2))
        d.setdefault(2, 100)
        d.setdefault(3, 200)
        self.assertEqual(d[2], 100)
        self.assertAlmostEqual(d[3], 3.4, places=4)
        d.update({4: 5, 5: 6})
        self.assertAlmostEqual(d[4], 5)
        self.assertAlmostEqual(d[5], 6)
        self.assertTrue(4 in d)
        pyd = dict(d.items())
        self.assertEqual(len(pyd), len(d))
        self.assertAlmostEqual(d.pop(4), 5)
        nelem = len(d)
        (k, v) = d.popitem()
        self.assertEqual(len(d), nelem - 1)
        self.assertTrue(k not in d)
        copied = d.copy()
        self.assertEqual(copied, d)
        self.assertEqual(list(copied.items()), list(d.items()))

    def test_copy_from_dict(self):
        if False:
            print('Hello World!')
        expect = {k: float(v) for (k, v) in zip(range(10), range(10, 20))}
        nbd = Dict.empty(int32, float64)
        for (k, v) in expect.items():
            nbd[k] = v
        got = dict(nbd)
        self.assertEqual(got, expect)

    def test_compiled(self):
        if False:
            while True:
                i = 10

        @njit
        def producer():
            if False:
                print('Hello World!')
            d = Dict.empty(int32, float64)
            d[1] = 1.23
            return d

        @njit
        def consumer(d):
            if False:
                print('Hello World!')
            return d[1]
        d = producer()
        val = consumer(d)
        self.assertEqual(val, 1.23)

    def test_gh7908(self):
        if False:
            for i in range(10):
                print('nop')
        d = Dict.empty(key_type=types.Tuple([types.uint32, types.uint32]), value_type=int64)
        d[1, 1] = 12345
        self.assertEqual(d[1, 1], d.get((1, 1)))

    def check_stringify(self, strfn, prefix=False):
        if False:
            for i in range(10):
                print('nop')
        nbd = Dict.empty(int32, int32)
        d = {}
        nbd[1] = 2
        d[1] = 2
        checker = self.assertIn if prefix else self.assertEqual
        checker(strfn(d), strfn(nbd))
        nbd[2] = 3
        d[2] = 3
        checker(strfn(d), strfn(nbd))
        for i in range(10, 20):
            nbd[i] = i + 1
            d[i] = i + 1
        checker(strfn(d), strfn(nbd))
        if prefix:
            self.assertTrue(strfn(nbd).startswith('DictType'))

    def test_repr(self):
        if False:
            while True:
                i = 10
        self.check_stringify(repr, prefix=True)

    def test_str(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_stringify(str)

class DictIterableCtor:

    def test_iterable_type_constructor(self):
        if False:
            while True:
                i = 10

        @njit
        def func1(a, b):
            if False:
                for i in range(10):
                    print('nop')
            d = Dict(zip(a, b))
            return d

        @njit
        def func2(a_, b):
            if False:
                return 10
            a = range(3)
            return Dict(zip(a, b))

        @njit
        def func3(a_, b):
            if False:
                for i in range(10):
                    print('nop')
            a = [0, 1, 2]
            return Dict(zip(a, b))

        @njit
        def func4(a, b):
            if False:
                i = 10
                return i + 15
            c = zip(a, b)
            return Dict(zip(a, zip(c, a)))

        @njit
        def func5(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return Dict(zip(zip(a, b), b))

        @njit
        def func6(items):
            if False:
                print('Hello World!')
            return Dict(items)

        @njit
        def func7(k, v):
            if False:
                return 10
            return Dict({k: v})

        @njit
        def func8(k, v):
            if False:
                print('Hello World!')
            d = Dict()
            d[k] = v
            return d

        def _get_dict(py_dict):
            if False:
                print('Hello World!')
            d = Dict()
            for (k, v) in py_dict.items():
                d[k] = v
            return d
        vals = ((func1, [(0, 1, 2), 'abc'], _get_dict({0: 'a', 1: 'b', 2: 'c'})), (func2, [(0, 1, 2), 'abc'], _get_dict({0: 'a', 1: 'b', 2: 'c'})), (func3, [(0, 1, 2), 'abc'], _get_dict({0: 'a', 1: 'b', 2: 'c'})), (func4, [(0, 1, 2), 'abc'], _get_dict({0: ((0, 'a'), 0), 1: ((1, 'b'), 1), 2: ((2, 'c'), 2)})), (func5, [(0, 1, 2), 'abc'], _get_dict({(0, 'a'): 'a', (1, 'b'): 'b', (2, 'c'): 'c'})), (func6, [((1, 'a'), (3, 'b'))], _get_dict({1: 'a', 3: 'b'})), (func1, ['key', _get_dict({1: 'abc'})], _get_dict({'k': 1})), (func8, ['key', _get_dict({1: 'abc'})], _get_dict({'key': _get_dict({1: 'abc'})})), (func8, ['key', List([1, 2, 3])], _get_dict({'key': List([1, 2, 3])})))
        for (func, args, expected) in vals:
            if self.jit_enabled:
                got = func(*args)
            else:
                got = func.py_func(*args)
            self.assertPreciseEqual(expected, got)

class TestDictIterableCtorJit(TestCase, DictIterableCtor):

    def setUp(self):
        if False:
            return 10
        self.jit_enabled = True

    def test_exception_no_iterable_arg(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def ctor():
            if False:
                for i in range(10):
                    print('nop')
            return Dict(3)
        msg = '.*No implementation of function.*'
        with self.assertRaisesRegex(TypingError, msg):
            ctor()

    def test_exception_dict_mapping(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def ctor():
            if False:
                while True:
                    i = 10
            return Dict({1: 2, 3: 4})
        msg = '.*No implementation of function.*'
        with self.assertRaisesRegex(TypingError, msg):
            ctor()

    def test_exception_setitem(self):
        if False:
            return 10

        @njit
        def ctor():
            if False:
                print('Hello World!')
            return Dict(((1, 'a'), (2, 'b', 3)))
        msg = '.*No implementation of function.*'
        with self.assertRaisesRegex(TypingError, msg):
            ctor()

class TestDictIterableCtorNoJit(TestCase, DictIterableCtor):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.jit_enabled = False

    def test_exception_nargs(self):
        if False:
            return 10
        msg = 'Dict expect at most 1 argument, got 2'
        with self.assertRaisesRegex(TypingError, msg):
            Dict(1, 2)

    def test_exception_mapping_ctor(self):
        if False:
            for i in range(10):
                print('nop')
        msg = '.*dict\\(mapping\\) is not supported.*'
        with self.assertRaisesRegex(TypingError, msg):
            Dict({1: 2})

    def test_exception_non_iterable_arg(self):
        if False:
            print('Hello World!')
        msg = '.*object is not iterable.*'
        with self.assertRaisesRegex(TypingError, msg):
            Dict(3)

    def test_exception_setitem(self):
        if False:
            print('Hello World!')
        msg = '.*dictionary update sequence element #1 has length 3.*'
        with self.assertRaisesRegex(ValueError, msg):
            Dict(((1, 'a'), (2, 'b', 3)))

class TestDictRefctTypes(MemoryLeakMixin, TestCase):

    def test_str_key(self):
        if False:
            return 10

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            d = Dict.empty(key_type=types.unicode_type, value_type=types.int32)
            d['123'] = 123
            d['321'] = 321
            return d
        d = foo()
        self.assertEqual(d['123'], 123)
        self.assertEqual(d['321'], 321)
        expect = {'123': 123, '321': 321}
        self.assertEqual(dict(d), expect)
        d['123'] = 231
        expect['123'] = 231
        self.assertEqual(d['123'], 231)
        self.assertEqual(dict(d), expect)
        nelem = 100
        for i in range(nelem):
            d[str(i)] = i
            expect[str(i)] = i
        for i in range(nelem):
            self.assertEqual(d[str(i)], i)
        self.assertEqual(dict(d), expect)

    def test_str_val(self):
        if False:
            while True:
                i = 10

        @njit
        def foo():
            if False:
                return 10
            d = Dict.empty(key_type=types.int32, value_type=types.unicode_type)
            d[123] = '123'
            d[321] = '321'
            return d
        d = foo()
        self.assertEqual(d[123], '123')
        self.assertEqual(d[321], '321')
        expect = {123: '123', 321: '321'}
        self.assertEqual(dict(d), expect)
        d[123] = '231'
        expect[123] = '231'
        self.assertEqual(dict(d), expect)
        nelem = 1
        for i in range(nelem):
            d[i] = str(i)
            expect[i] = str(i)
        for i in range(nelem):
            self.assertEqual(d[i], str(i))
        self.assertEqual(dict(d), expect)

    def test_str_key_array_value(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(123)
        d = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
        expect = []
        expect.append(np.random.random(10))
        d['mass'] = expect[-1]
        expect.append(np.random.random(20))
        d['velocity'] = expect[-1]
        for i in range(100):
            expect.append(np.random.random(i))
            d[str(i)] = expect[-1]
        self.assertEqual(len(d), len(expect))
        self.assertPreciseEqual(d['mass'], expect[0])
        self.assertPreciseEqual(d['velocity'], expect[1])
        for (got, exp) in zip(d.values(), expect):
            self.assertPreciseEqual(got, exp)
        self.assertTrue('mass' in d)
        self.assertTrue('velocity' in d)
        del d['mass']
        self.assertFalse('mass' in d)
        del d['velocity']
        self.assertFalse('velocity' in d)
        del expect[0:2]
        for i in range(90):
            (k, v) = d.popitem()
            w = expect.pop()
            self.assertPreciseEqual(v, w)
        expect.append(np.random.random(10))
        d['last'] = expect[-1]
        for (got, exp) in zip(d.values(), expect):
            self.assertPreciseEqual(got, exp)

    def test_dict_of_dict_int_keyval(self):
        if False:
            i = 10
            return i + 15

        def inner_numba_dict():
            if False:
                print('Hello World!')
            d = Dict.empty(key_type=types.intp, value_type=types.intp)
            return d
        d = Dict.empty(key_type=types.intp, value_type=types.DictType(types.intp, types.intp))

        def usecase(d, make_inner_dict):
            if False:
                print('Hello World!')
            for i in range(100):
                mid = make_inner_dict()
                for j in range(i + 1):
                    mid[j] = j * 10000
                d[i] = mid
            return d
        got = usecase(d, inner_numba_dict)
        expect = usecase({}, dict)
        self.assertIsInstance(expect, dict)
        self.assertEqual(dict(got), expect)
        for where in [12, 3, 6, 8, 10]:
            del got[where]
            del expect[where]
            self.assertEqual(dict(got), expect)

    def test_dict_of_dict_npm(self):
        if False:
            i = 10
            return i + 15
        inner_dict_ty = types.DictType(types.intp, types.intp)

        @njit
        def inner_numba_dict():
            if False:
                while True:
                    i = 10
            d = Dict.empty(key_type=types.intp, value_type=types.intp)
            return d

        @njit
        def foo(count):
            if False:
                while True:
                    i = 10
            d = Dict.empty(key_type=types.intp, value_type=inner_dict_ty)
            for i in range(count):
                d[i] = inner_numba_dict()
                for j in range(i + 1):
                    d[i][j] = j
            return d
        d = foo(100)
        ct = 0
        for (k, dd) in d.items():
            ct += 1
            self.assertEqual(len(dd), k + 1)
            for (kk, vv) in dd.items():
                self.assertEqual(kk, vv)
        self.assertEqual(ct, 100)

    def test_delitem(self):
        if False:
            for i in range(10):
                print('nop')
        d = Dict.empty(types.int64, types.unicode_type)
        d[1] = 'apple'

        @njit
        def foo(x, k):
            if False:
                for i in range(10):
                    print('nop')
            del x[1]
        foo(d, 1)
        self.assertEqual(len(d), 0)
        self.assertFalse(d)

    def test_getitem_return_type(self):
        if False:
            print('Hello World!')
        d = Dict.empty(types.int64, types.int64[:])
        d[1] = np.arange(10, dtype=np.int64)

        @njit
        def foo(d):
            if False:
                return 10
            d[1] += 100
            return d[1]
        foo(d)
        retty = foo.nopython_signatures[0].return_type
        self.assertIsInstance(retty, types.Array)
        self.assertNotIsInstance(retty, types.Optional)
        self.assertPreciseEqual(d[1], np.arange(10, dtype=np.int64) + 100)

    def test_storage_model_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        dct = Dict()
        ref = [('a', True, 'a'), ('b', False, 'b'), ('c', False, 'c')]
        for x in ref:
            dct[x] = x
        for (i, x) in enumerate(ref):
            self.assertEqual(dct[x], x)

class TestDictForbiddenTypes(TestCase):

    def assert_disallow(self, expect, callable):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypingError) as raises:
            callable()
        msg = str(raises.exception)
        self.assertIn(expect, msg)

    def assert_disallow_key(self, ty):
        if False:
            i = 10
            return i + 15
        msg = '{} as key is forbidden'.format(ty)
        self.assert_disallow(msg, lambda : Dict.empty(ty, types.intp))

        @njit
        def foo():
            if False:
                return 10
            Dict.empty(ty, types.intp)
        self.assert_disallow(msg, foo)

    def assert_disallow_value(self, ty):
        if False:
            i = 10
            return i + 15
        msg = '{} as value is forbidden'.format(ty)
        self.assert_disallow(msg, lambda : Dict.empty(types.intp, ty))

        @njit
        def foo():
            if False:
                return 10
            Dict.empty(types.intp, ty)
        self.assert_disallow(msg, foo)

    def test_disallow_list(self):
        if False:
            i = 10
            return i + 15
        self.assert_disallow_key(types.List(types.intp))
        self.assert_disallow_value(types.List(types.intp))

    def test_disallow_set(self):
        if False:
            print('Hello World!')
        self.assert_disallow_key(types.Set(types.intp))
        self.assert_disallow_value(types.Set(types.intp))

class TestDictInferred(TestCase):

    def test_simple_literal(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            d = Dict()
            d[123] = 321
            return d
        (k, v) = (123, 321)
        d = foo()
        self.assertEqual(dict(d), {k: v})
        self.assertEqual(typeof(d).key_type, typeof(k))
        self.assertEqual(typeof(d).value_type, typeof(v))

    def test_simple_args(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(k, v):
            if False:
                return 10
            d = Dict()
            d[k] = v
            return d
        (k, v) = (123, 321)
        d = foo(k, v)
        self.assertEqual(dict(d), {k: v})
        self.assertEqual(typeof(d).key_type, typeof(k))
        self.assertEqual(typeof(d).value_type, typeof(v))

    def test_simple_upcast(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo(k, v, w):
            if False:
                i = 10
                return i + 15
            d = Dict()
            d[k] = v
            d[k] = w
            return d
        (k, v, w) = (123, 32.1, 321)
        d = foo(k, v, w)
        self.assertEqual(dict(d), {k: w})
        self.assertEqual(typeof(d).key_type, typeof(k))
        self.assertEqual(typeof(d).value_type, typeof(v))

    def test_conflicting_value_type(self):
        if False:
            print('Hello World!')

        @njit
        def foo(k, v, w):
            if False:
                while True:
                    i = 10
            d = Dict()
            d[k] = v
            d[k] = w
            return d
        (k, v, w) = (123, 321, 32.1)
        with self.assertRaises(TypingError) as raises:
            foo(k, v, w)
        self.assertIn('cannot safely cast float64 to {}'.format(typeof(v)), str(raises.exception))

    def test_conflicting_key_type(self):
        if False:
            while True:
                i = 10

        @njit
        def foo(k, h, v):
            if False:
                print('Hello World!')
            d = Dict()
            d[k] = v
            d[h] = v
            return d
        (k, h, v) = (123, 123.1, 321)
        with self.assertRaises(TypingError) as raises:
            foo(k, h, v)
        self.assertIn('cannot safely cast float64 to {}'.format(typeof(v)), str(raises.exception))

    def test_conflict_key_type_non_number(self):
        if False:
            print('Hello World!')

        @njit
        def foo(k1, v1, k2):
            if False:
                return 10
            d = Dict()
            d[k1] = v1
            return (d, d[k2])
        k1 = (np.int8(1), np.int8(2))
        k2 = (np.int32(1), np.int32(2))
        v1 = np.intp(123)
        with warnings.catch_warnings(record=True) as w:
            (d, dk2) = foo(k1, v1, k2)
        self.assertEqual(len(w), 1)
        msg = 'unsafe cast from UniTuple(int32 x 2) to UniTuple(int8 x 2)'
        self.assertIn(msg, str(w[0]))
        keys = list(d.keys())
        self.assertEqual(keys[0], (1, 2))
        self.assertEqual(dk2, d[np.int32(1), np.int32(2)])

    def test_ifelse_filled_both_branches(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(k, v):
            if False:
                for i in range(10):
                    print('nop')
            d = Dict()
            if k:
                d[k] = v
            else:
                d[57005] = v + 1
            return d
        (k, v) = (123, 321)
        d = foo(k, v)
        self.assertEqual(dict(d), {k: v})
        (k, v) = (0, 0)
        d = foo(k, v)
        self.assertEqual(dict(d), {57005: v + 1})

    def test_ifelse_empty_one_branch(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(k, v):
            if False:
                print('Hello World!')
            d = Dict()
            if k:
                d[k] = v
            return d
        (k, v) = (123, 321)
        d = foo(k, v)
        self.assertEqual(dict(d), {k: v})
        (k, v) = (0, 0)
        d = foo(k, v)
        self.assertEqual(dict(d), {})
        self.assertEqual(typeof(d).key_type, typeof(k))
        self.assertEqual(typeof(d).value_type, typeof(v))

    def test_loop(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(ks, vs):
            if False:
                for i in range(10):
                    print('nop')
            d = Dict()
            for (k, v) in zip(ks, vs):
                d[k] = v
            return d
        vs = list(range(4))
        ks = list(map(lambda x: x + 100, vs))
        d = foo(ks, vs)
        self.assertEqual(dict(d), dict(zip(ks, vs)))

    def test_unused(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                print('Hello World!')
            d = Dict()
            return d
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('imprecise type', str(raises.exception))

    def test_define_after_use(self):
        if False:
            print('Hello World!')

        @njit
        def foo(define):
            if False:
                i = 10
                return i + 15
            d = Dict()
            ct = len(d)
            for (k, v) in d.items():
                ct += v
            if define:
                d[1] = 2
            return (ct, d, len(d))
        (ct, d, n) = foo(True)
        self.assertEqual(ct, 0)
        self.assertEqual(n, 1)
        self.assertEqual(dict(d), {1: 2})
        (ct, d, n) = foo(False)
        self.assertEqual(ct, 0)
        self.assertEqual(dict(d), {})
        self.assertEqual(n, 0)

    def test_dict_of_dict(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo(k1, k2, v):
            if False:
                for i in range(10):
                    print('nop')
            d = Dict()
            z1 = Dict()
            z1[k1 + 1] = v + k1
            z2 = Dict()
            z2[k2 + 2] = v + k2
            d[k1] = z1
            d[k2] = z2
            return d
        (k1, k2, v) = (100, 200, 321)
        d = foo(k1, k2, v)
        self.assertEqual(dict(d), {k1: {k1 + 1: k1 + v}, k2: {k2 + 2: k2 + v}})

    def test_comprehension_basic(self):
        if False:
            while True:
                i = 10

        @njit
        def foo():
            if False:
                while True:
                    i = 10
            return {i: 2 * i for i in range(10)}
        self.assertEqual(foo(), foo.py_func())

    def test_comprehension_basic_mixed_type(self):
        if False:
            print('Hello World!')

        @njit
        def foo():
            if False:
                i = 10
                return i + 15
            return {i: float(j) for (i, j) in zip(range(10), range(10, 0, -1))}
        self.assertEqual(foo(), foo.py_func())

    def test_comprehension_involved(self):
        if False:
            return 10

        @njit
        def foo():
            if False:
                return 10
            a = {0: 'A', 1: 'B', 2: 'C'}
            return {3 + i: a[i] for i in range(3)}
        self.assertEqual(foo(), foo.py_func())

    def test_comprehension_fail_mixed_type(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            a = {0: 'A', 1: 'B', 2: 1j}
            return {3 + i: a[i] for i in range(3)}
        with self.assertRaises(TypingError) as e:
            foo()
        excstr = str(e.exception)
        self.assertIn('Cannot cast complex128 to unicode_type', excstr)

class TestNonCompiledInfer(TestCase):

    def test_check_untyped_dict_ops(self):
        if False:
            return 10
        d = Dict()
        self.assertFalse(d._typed)
        self.assertEqual(len(d), 0)
        self.assertEqual(str(d), str({}))
        self.assertEqual(list(iter(d)), [])
        with self.assertRaises(KeyError) as raises:
            d[1]
        self.assertEqual(str(raises.exception), str(KeyError(1)))
        with self.assertRaises(KeyError) as raises:
            del d[1]
        self.assertEqual(str(raises.exception), str(KeyError(1)))
        with self.assertRaises(KeyError):
            d.pop(1)
        self.assertEqual(str(raises.exception), str(KeyError(1)))
        self.assertIs(d.pop(1, None), None)
        self.assertIs(d.get(1), None)
        with self.assertRaises(KeyError) as raises:
            d.popitem()
        self.assertEqual(str(raises.exception), str(KeyError('dictionary is empty')))
        with self.assertRaises(TypeError) as raises:
            d.setdefault(1)
        self.assertEqual(str(raises.exception), str(TypeError('invalid operation on untyped dictionary')))
        self.assertFalse(1 in d)
        self.assertFalse(d._typed)

    def test_getitem(self):
        if False:
            i = 10
            return i + 15
        d = Dict()
        d[1] = 2
        self.assertTrue(d._typed)
        self.assertEqual(d[1], 2)

    def test_setdefault(self):
        if False:
            print('Hello World!')
        d = Dict()
        d.setdefault(1, 2)
        self.assertTrue(d._typed)
        self.assertEqual(d[1], 2)

@jitclass(spec=[('a', types.intp)])
class Bag(object):

    def __init__(self, a):
        if False:
            return 10
        self.a = a

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.a)

class TestDictWithJitclass(TestCase):

    def test_jitclass_as_value(self):
        if False:
            return 10

        @njit
        def foo(x):
            if False:
                return 10
            d = Dict()
            d[0] = x
            d[1] = Bag(101)
            return d
        d = foo(Bag(a=100))
        self.assertEqual(d[0].a, 100)
        self.assertEqual(d[1].a, 101)

class TestNoJit(TestCase):
    """Exercise dictionary creation with JIT disabled. """

    def test_dict_create_no_jit_using_new_dict(self):
        if False:
            while True:
                i = 10
        with override_config('DISABLE_JIT', True):
            with forbid_codegen():
                d = dictobject.new_dict(int32, float32)
                self.assertEqual(type(d), dict)

    def test_dict_create_no_jit_using_Dict(self):
        if False:
            for i in range(10):
                print('nop')
        with override_config('DISABLE_JIT', True):
            with forbid_codegen():
                d = Dict()
                self.assertEqual(type(d), dict)

    def test_dict_create_no_jit_using_empty(self):
        if False:
            print('Hello World!')
        with override_config('DISABLE_JIT', True):
            with forbid_codegen():
                d = Dict.empty(types.int32, types.float32)
                self.assertEqual(type(d), dict)

class TestDictIterator(TestCase):

    def test_dict_iterator(self):
        if False:
            print('Hello World!')

        @njit
        def fun1():
            if False:
                for i in range(10):
                    print('nop')
            dd = Dict.empty(key_type=types.intp, value_type=types.intp)
            dd[0] = 10
            dd[1] = 20
            dd[2] = 30
            return (list(dd.keys()), list(dd.values()))

        @njit
        def fun2():
            if False:
                i = 10
                return i + 15
            dd = Dict.empty(key_type=types.intp, value_type=types.intp)
            dd[4] = 77
            dd[5] = 88
            dd[6] = 99
            return (list(dd.keys()), list(dd.values()))
        res1 = fun1()
        res2 = fun2()
        self.assertEqual([0, 1, 2], res1[0])
        self.assertEqual([10, 20, 30], res1[1])
        self.assertEqual([4, 5, 6], res2[0])
        self.assertEqual([77, 88, 99], res2[1])

class TestTypedDictInitialValues(MemoryLeakMixin, TestCase):
    """Tests that typed dictionaries carry their initial value if present"""

    def test_homogeneous_and_literal(self):
        if False:
            i = 10
            return i + 15

        def bar(d):
            if False:
                return 10
            ...

        @overload(bar)
        def ol_bar(d):
            if False:
                while True:
                    i = 10
            if d.initial_value is None:
                return lambda d: literally(d)
            self.assertTrue(isinstance(d, types.DictType))
            self.assertEqual(d.initial_value, {'a': 1, 'b': 2, 'c': 3})
            self.assertEqual(hasattr(d, 'literal_value'), False)
            return lambda d: d

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            x = {'a': 1, 'b': 2, 'c': 3}
            bar(x)
        foo()

    def test_heterogeneous_but_castable_to_homogeneous(self):
        if False:
            i = 10
            return i + 15

        def bar(d):
            if False:
                i = 10
                return i + 15
            ...

        @overload(bar)
        def ol_bar(d):
            if False:
                i = 10
                return i + 15
            self.assertTrue(isinstance(d, types.DictType))
            self.assertEqual(d.initial_value, None)
            self.assertEqual(hasattr(d, 'literal_value'), False)
            return lambda d: d

        @njit
        def foo():
            if False:
                return 10
            x = {'a': 1j, 'b': 2, 'c': 3}
            bar(x)
        foo()

    def test_heterogeneous_but_not_castable_to_homogeneous(self):
        if False:
            i = 10
            return i + 15

        def bar(d):
            if False:
                while True:
                    i = 10
            ...

        @overload(bar)
        def ol_bar(d):
            if False:
                i = 10
                return i + 15
            a = {'a': 1, 'b': 2j, 'c': 3}

            def specific_ty(z):
                if False:
                    return 10
                return types.literal(z) if types.maybe_literal(z) else typeof(z)
            expected = {types.literal(x): specific_ty(y) for (x, y) in a.items()}
            self.assertTrue(isinstance(d, types.LiteralStrKeyDict))
            self.assertEqual(d.literal_value, expected)
            self.assertEqual(hasattr(d, 'initial_value'), False)
            return lambda d: d

        @njit
        def foo():
            if False:
                return 10
            x = {'a': 1, 'b': 2j, 'c': 3}
            bar(x)
        foo()

    def test_mutation_not_carried(self):
        if False:
            for i in range(10):
                print('nop')

        def bar(d):
            if False:
                i = 10
                return i + 15
            ...

        @overload(bar)
        def ol_bar(d):
            if False:
                for i in range(10):
                    print('nop')
            if d.initial_value is None:
                return lambda d: literally(d)
            self.assertTrue(isinstance(d, types.DictType))
            self.assertEqual(d.initial_value, {'a': 1, 'b': 2, 'c': 3})
            return lambda d: d

        @njit
        def foo():
            if False:
                return 10
            x = {'a': 1, 'b': 2, 'c': 3}
            x['d'] = 4
            bar(x)
        foo()

    def test_mutation_not_carried_single_function(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def nop(*args):
            if False:
                for i in range(10):
                    print('nop')
            pass
        for (fn, iv) in ((nop, None), (literally, {'a': 1, 'b': 2, 'c': 3})):

            @njit
            def baz(x):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def bar(z):
                if False:
                    i = 10
                    return i + 15
                pass

            @overload(bar)
            def ol_bar(z):
                if False:
                    return 10

                def impl(z):
                    if False:
                        print('Hello World!')
                    fn(z)
                    baz(z)
                return impl

            @njit
            def foo():
                if False:
                    return 10
                x = {'a': 1, 'b': 2, 'c': 3}
                bar(x)
                x['d'] = 4
                return x
            foo()
            larg = baz.signatures[0][0]
            self.assertEqual(larg.initial_value, iv)

    def test_unify_across_function_call(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def bar(x):
            if False:
                while True:
                    i = 10
            o = {1: 2}
            if x:
                o = {2: 3}
            return o

        @njit
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            if x:
                d = {3: 4}
            else:
                d = bar(x)
            return d
        e1 = Dict()
        e1[3] = 4
        e2 = Dict()
        e2[1] = 2
        self.assertEqual(foo(True), e1)
        self.assertEqual(foo(False), e2)

class TestLiteralStrKeyDict(MemoryLeakMixin, TestCase):
    """ Tests for dictionaries with string keys that can map to anything!"""

    def test_basic_const_lowering_boxing(self):
        if False:
            print('Hello World!')

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            return (ld['a'], ld['b'], ld['c'])
        self.assertEqual(foo(), (1, 2j, 'd'))

    def test_basic_nonconst_in_scope(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            y = x + 5
            e = True if y > 2 else False
            ld = {'a': 1, 'b': 2j, 'c': 'd', 'non_const': e}
            return ld['non_const']
        self.assertTrue(foo(34))
        self.assertFalse(foo(-100))

    def test_basic_nonconst_freevar(self):
        if False:
            for i in range(10):
                print('nop')
        e = 5

        def bar(x):
            if False:
                while True:
                    i = 10
            pass

        @overload(bar)
        def ol_bar(x):
            if False:
                print('Hello World!')
            self.assertEqual(x.literal_value, {types.literal('a'): types.literal(1), types.literal('b'): typeof(2j), types.literal('c'): types.literal('d'), types.literal('d'): types.literal(5)})

            def impl(x):
                if False:
                    print('Hello World!')
                pass
            return impl

        @njit
        def foo():
            if False:
                i = 10
                return i + 15
            ld = {'a': 1, 'b': 2j, 'c': 'd', 'd': e}
            bar(ld)
        foo()

    def test_literal_value(self):
        if False:
            return 10

        def bar(x):
            if False:
                while True:
                    i = 10
            pass

        @overload(bar)
        def ol_bar(x):
            if False:
                print('Hello World!')
            self.assertEqual(x.literal_value, {types.literal('a'): types.literal(1), types.literal('b'): typeof(2j), types.literal('c'): types.literal('d')})

            def impl(x):
                if False:
                    while True:
                        i = 10
                pass
            return impl

        @njit
        def foo():
            if False:
                print('Hello World!')
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            bar(ld)
        foo()

    def test_list_and_array_as_value(self):
        if False:
            while True:
                i = 10

        def bar(x):
            if False:
                return 10
            pass

        @overload(bar)
        def ol_bar(x):
            if False:
                print('Hello World!')
            self.assertEqual(x.literal_value, {types.literal('a'): types.literal(1), types.literal('b'): types.List(types.intp, initial_value=[1, 2, 3]), types.literal('c'): typeof(np.zeros(5))})

            def impl(x):
                if False:
                    i = 10
                    return i + 15
                pass
            return impl

        @njit
        def foo():
            if False:
                while True:
                    i = 10
            b = [1, 2, 3]
            ld = {'a': 1, 'b': b, 'c': np.zeros(5)}
            bar(ld)
        foo()

    def test_repeated_key_literal_value(self):
        if False:
            i = 10
            return i + 15

        def bar(x):
            if False:
                print('Hello World!')
            pass

        @overload(bar)
        def ol_bar(x):
            if False:
                i = 10
                return i + 15
            self.assertEqual(x.literal_value, {types.literal('a'): types.literal('aaaa'), types.literal('b'): typeof(2j), types.literal('c'): types.literal('d')})

            def impl(x):
                if False:
                    for i in range(10):
                        print('nop')
                pass
            return impl

        @njit
        def foo():
            if False:
                print('Hello World!')
            ld = {'a': 1, 'a': 10, 'b': 2j, 'c': 'd', 'a': 'aaaa'}
            bar(ld)
        foo()

    def test_read_only(self):
        if False:
            while True:
                i = 10

        def _len():
            if False:
                print('Hello World!')
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            return len(ld)

        def static_getitem():
            if False:
                return 10
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            return ld['b']

        def contains():
            if False:
                print('Hello World!')
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            return ('b' in ld, 'f' in ld)

        def copy():
            if False:
                for i in range(10):
                    print('nop')
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            new = ld.copy()
            return ld == new
        rdonlys = (_len, static_getitem, contains, copy)
        for test in rdonlys:
            with self.subTest(test.__name__):
                self.assertPreciseEqual(njit(test)(), test())

    def test_mutation_failure(self):
        if False:
            i = 10
            return i + 15

        def setitem():
            if False:
                return 10
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            ld['a'] = 12

        def delitem():
            if False:
                while True:
                    i = 10
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            del ld['a']

        def popitem():
            if False:
                return 10
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            ld.popitem()

        def pop():
            if False:
                return 10
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            ld.pop()

        def clear():
            if False:
                while True:
                    i = 10
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            ld.clear()

        def setdefault():
            if False:
                return 10
            ld = {'a': 1, 'b': 2j, 'c': 'd'}
            ld.setdefault('f', 1)
        illegals = (setitem, delitem, popitem, pop, clear, setdefault)
        for test in illegals:
            with self.subTest(test.__name__):
                with self.assertRaises(TypingError) as raises:
                    njit(test)()
                expect = 'Cannot mutate a literal dictionary'
                self.assertIn(expect, str(raises.exception))

    def test_get(self):
        if False:
            while True:
                i = 10

        @njit
        def get(x):
            if False:
                i = 10
                return i + 15
            ld = {'a': 2j, 'c': 'd'}
            return ld.get(x)

        @njit
        def getitem(x):
            if False:
                while True:
                    i = 10
            ld = {'a': 2j, 'c': 'd'}
            return ld[x]
        for test in (get, getitem):
            with self.subTest(test.__name__):
                with self.assertRaises(TypingError) as raises:
                    test('a')
                expect = 'Cannot get{item}() on a literal dictionary'
                self.assertIn(expect, str(raises.exception))

    def test_dict_keys(self):
        if False:
            while True:
                i = 10

        @njit
        def foo():
            if False:
                while True:
                    i = 10
            ld = {'a': 2j, 'c': 'd'}
            return [x for x in ld.keys()]
        self.assertEqual(foo(), ['a', 'c'])

    def test_dict_values(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo():
            if False:
                i = 10
                return i + 15
            ld = {'a': 2j, 'c': 'd'}
            return ld.values()
        self.assertEqual(foo(), (2j, 'd'))

    def test_dict_items(self):
        if False:
            print('Hello World!')

        @njit
        def foo():
            if False:
                print('Hello World!')
            ld = {'a': 2j, 'c': 'd', 'f': np.zeros(5)}
            return ld.items()
        self.assertPreciseEqual(foo(), (('a', 2j), ('c', 'd'), ('f', np.zeros(5))))

    def test_dict_return(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            ld = {'a': 2j, 'c': 'd'}
            return ld
        with self.assertRaises(TypeError) as raises:
            foo()
        excstr = str(raises.exception)
        self.assertIn('cannot convert native LiteralStrKey', excstr)

    def test_dict_unify(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            if x + 7 > 4:
                a = {'a': 2j, 'c': 'd', 'e': np.zeros(4)}
            else:
                a = {'a': 5j, 'c': 'CAT', 'e': np.zeros((5,))}
            return a['c']
        self.assertEqual(foo(100), 'd')
        self.assertEqual(foo(-100), 'CAT')
        self.assertEqual(foo(100), foo.py_func(100))
        self.assertEqual(foo(-100), foo.py_func(-100))

    def test_dict_not_unify(self):
        if False:
            while True:
                i = 10

        @njit
        def key_mismatch(x):
            if False:
                print('Hello World!')
            if x + 7 > 4:
                a = {'BAD_KEY': 2j, 'c': 'd', 'e': np.zeros(4)}
            else:
                a = {'a': 5j, 'c': 'CAT', 'e': np.zeros((5,))}
            py310_defeat1 = 1
            py310_defeat2 = 2
            py310_defeat3 = 3
            py310_defeat4 = 4
            return a['a']
        with self.assertRaises(TypingError) as raises:
            key_mismatch(100)
        self.assertIn('Cannot unify LiteralStrKey', str(raises.exception))

        @njit
        def value_type_mismatch(x):
            if False:
                return 10
            if x + 7 > 4:
                a = {'a': 2j, 'c': 'd', 'e': np.zeros((4, 3))}
            else:
                a = {'a': 5j, 'c': 'CAT', 'e': np.zeros((5,))}
            py310_defeat1 = 1
            py310_defeat2 = 2
            py310_defeat3 = 3
            py310_defeat4 = 4
            return a['a']
        with self.assertRaises(TypingError) as raises:
            value_type_mismatch(100)
        self.assertIn('Cannot unify LiteralStrKey', str(raises.exception))

    def test_dict_value_coercion(self):
        if False:
            return 10
        p = {(np.int32, np.int32): types.DictType, (np.int32, np.int8): types.DictType, (np.complex128, np.int32): types.DictType, (np.int32, np.complex128): types.LiteralStrKeyDict, (np.int32, np.array): types.LiteralStrKeyDict, (np.array, np.int32): types.LiteralStrKeyDict, (np.int8, np.int32): types.LiteralStrKeyDict, (np.int64, np.float64): types.LiteralStrKeyDict}

        def bar(x):
            if False:
                i = 10
                return i + 15
            pass
        for (dts, container) in p.items():

            @overload(bar)
            def ol_bar(x):
                if False:
                    i = 10
                    return i + 15
                self.assertTrue(isinstance(x, container))

                def impl(x):
                    if False:
                        print('Hello World!')
                    pass
                return impl
            (ty1, ty2) = dts

            @njit
            def foo():
                if False:
                    for i in range(10):
                        print('nop')
                d = {'a': ty1(1), 'b': ty2(2)}
                bar(d)
            foo()

    def test_build_map_op_code(self):
        if False:
            for i in range(10):
                print('nop')

        def bar(x):
            if False:
                while True:
                    i = 10
            pass

        @overload(bar)
        def ol_bar(x):
            if False:
                while True:
                    i = 10

            def impl(x):
                if False:
                    return 10
                pass
            return impl

        @njit
        def foo():
            if False:
                i = 10
                return i + 15
            a = {'a': {'b1': 10, 'b2': 'string'}}
            bar(a)
        foo()

    def test_dict_as_arg(self):
        if False:
            print('Hello World!')

        @njit
        def bar(fake_kwargs=None):
            if False:
                return 10
            if fake_kwargs is not None:
                fake_kwargs['d'][:] += 10

        @njit
        def foo():
            if False:
                print('Hello World!')
            a = 1
            b = 2j
            c = 'string'
            d = np.zeros(3)
            e = {'a': a, 'b': b, 'c': c, 'd': d}
            bar(fake_kwargs=e)
            return e['d']
        np.testing.assert_allclose(foo(), np.ones(3) * 10)

    def test_dict_with_single_literallist_value(self):
        if False:
            return 10

        @njit
        def foo():
            if False:
                while True:
                    i = 10
            z = {'A': [lambda a: 2 * a, 'B']}
            return z['A'][0](5)
        self.assertPreciseEqual(foo(), foo.py_func())

    def test_tuple_not_in_mro(self):
        if False:
            print('Hello World!')

        def bar(x):
            if False:
                print('Hello World!')
            pass

        @overload(bar)
        def ol_bar(x):
            if False:
                print('Hello World!')
            self.assertFalse(isinstance(x, types.BaseTuple))
            self.assertTrue(isinstance(x, types.LiteralStrKeyDict))
            return lambda x: ...

        @njit
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            d = {'a': 1, 'b': 'c'}
            bar(d)
        foo()

    def test_const_key_not_in_dict(self):
        if False:
            print('Hello World!')

        @njit
        def foo():
            if False:
                return 10
            a = {'not_a': 2j, 'c': 'd', 'e': np.zeros(4)}
            return a['a']
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn("Key 'a' is not in dict.", str(raises.exception))

    def test_uncommon_identifiers(self):
        if False:
            i = 10
            return i + 15

        @njit
        def foo():
            if False:
                print('Hello World!')
            d = {'0': np.ones(5), '1': 4}
            return len(d)
        self.assertPreciseEqual(foo(), foo.py_func())

        @njit
        def bar():
            if False:
                while True:
                    i = 10
            d = {'+': np.ones(5), 'x--': 4}
            return len(d)
        self.assertPreciseEqual(bar(), bar.py_func())

    def test_update_error(self):
        if False:
            print('Hello World!')

        @njit
        def foo():
            if False:
                i = 10
                return i + 15
            d1 = {'a': 2, 'b': 4, 'c': 'a'}
            d1.update({'x': 3})
            return d1
        with self.assertRaises(TypingError) as raises:
            foo()
        self.assertIn('Cannot mutate a literal dictionary', str(raises.exception))
if __name__ == '__main__':
    unittest.main()
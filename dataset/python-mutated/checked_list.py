from __static__ import CheckedList
from unittest import skip, skipIf
from cinderx.static import SEQ_CHECKED_LIST, SEQ_SUBSCR_UNCHECKED
from .common import bad_ret_type, StaticTestBase, type_mismatch
try:
    import cinderjit
except ImportError:
    cinderjit = None

class CheckedListTests(StaticTestBase):

    def test_checked_list(self):
        if False:
            print('Hello World!')
        x = CheckedList[int]()
        x.append(1)
        self.assertEqual(repr(x), '[1]')
        self.assertEqual(CheckedList[int].__module__, '__static__')

    def test_checked_list_append_bad_type(self):
        if False:
            i = 10
            return i + 15
        x = CheckedList[int]()
        x.append(42)
        self.assertEqual(x[0], 42)
        with self.assertRaisesRegex(TypeError, 'int'):
            x.append('A')

    def test_checked_list_free_list(self):
        if False:
            return 10
        t1 = CheckedList[str]
        t2 = CheckedList[str]
        x = t1()
        x_id1 = id(x)
        del x
        x = t2()
        x_id2 = id(x)
        self.assertEqual(x_id1, x_id2)

    def test_checked_list_insert(self):
        if False:
            for i in range(10):
                print('nop')
        x = CheckedList[int]()
        x.insert(0, 12)
        x.insert(0, 23)
        self.assertEqual(repr(x), '[23, 12]')
        with self.assertRaisesRegex(TypeError, 'argument 2 expected int'):
            x.insert(1, 'ASD')

    def test_checked_list_reversed(self):
        if False:
            while True:
                i = 10
        x = CheckedList[int]()
        x.append(12)
        x.append(23)
        y = x.__reversed__()
        self.assertEqual(repr(type(y)), "<class 'list_reverseiterator'>")
        self.assertEqual(repr(list(y)), '[23, 12]')

    def test_checked_list_sizeof(self):
        if False:
            i = 10
            return i + 15
        x = CheckedList[int]()
        x.append(12)
        x.append(23)
        self.assertGreater(x.__sizeof__(), 20)
        self.assertLess(x.__sizeof__(), 100)

    def test_checked_list_clear(self):
        if False:
            for i in range(10):
                print('nop')
        x = CheckedList[int]()
        x.append(12)
        x.append(23)
        self.assertEqual(repr(x), '[12, 23]')
        x.clear()
        self.assertEqual(repr(x), '[]')
        self.assertEqual(str(type(x)), "<class '__static__.chklist[int]'>")

    def test_checked_list_copy(self):
        if False:
            print('Hello World!')
        x = CheckedList[int]()
        x.append(12)
        x.append(23)
        self.assertEqual(repr(x), '[12, 23]')
        y = x.copy()
        y.append(34)
        self.assertEqual(repr(x), '[12, 23]')
        self.assertEqual(repr(y), '[12, 23, 34]')
        self.assertEqual(str(type(y)), "<class '__static__.chklist[int]'>")

        class C:
            x: int = 0
        clist = CheckedList[C]()
        clist.append(C())
        clist_copy = clist.copy()
        self.assertEqual(clist[0].x, 0)
        clist_copy[0].x = 1
        self.assertEqual(clist[0].x, 1)

    def test_checked_list_extend(self):
        if False:
            i = 10
            return i + 15
        x = CheckedList[int]()
        x.append(3)
        x.extend([1, 2])
        self.assertEqual(repr(x), '[3, 1, 2]')
        self.assertEqual(repr(type(x)), "<class '__static__.chklist[int]'>")

    def test_checked_list_extend_type_error(self):
        if False:
            i = 10
            return i + 15
        x = CheckedList[int]()
        x.append(3)
        with self.assertRaisesRegex(TypeError, "bad value 'str' for chklist\\[int\\]"):
            x.extend([1, 'hi'])
        self.assertEqual(repr(x), '[3]')

    def test_checked_list_extend_with_non_list(self):
        if False:
            i = 10
            return i + 15
        x = CheckedList[int]()
        d = {1: 2, 2: 3}
        x.extend(d.keys())
        self.assertEqual(repr(x), '[1, 2]')
        d['a'] = 4
        with self.assertRaisesRegex(TypeError, "bad value 'str' for chklist\\[int\\]"):
            x.extend(d.keys())
        self.assertEqual(repr(x), '[1, 2, 1, 2]')

    def test_checked_list_pop(self):
        if False:
            print('Hello World!')
        x = CheckedList[int]()
        x.extend([1, 2, 3, 4])
        self.assertEqual(x.pop(), 4)
        self.assertEqual(x.pop(), 3)
        self.assertEqual(x.pop(), 2)
        self.assertEqual(x.pop(), 1)
        with self.assertRaises(IndexError):
            x.pop()

    def test_checked_list_pop_compile(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__ import CheckedList\n            def foo(cl: CheckedList[int]) -> int:\n                return cl.pop()\n        '
        with self.in_module(codestr) as mod:
            f = mod.foo
            self.assertInBytecode(f, 'INVOKE_FUNCTION')
            self.assertNotInBytecode(f, 'CAST')
            cl = CheckedList[int]([1, 2, 3])
            self.assertEqual(f(cl), 3)

    def test_checked_list_remove(self):
        if False:
            while True:
                i = 10
        x = CheckedList[int]()
        x.extend([1, 2, 3, 1, 2, 3])
        x.remove(1)
        self.assertEqual(repr(x), '[2, 3, 1, 2, 3]')
        x.remove(1)
        self.assertEqual(repr(x), '[2, 3, 2, 3]')
        with self.assertRaises(ValueError):
            x.remove(1)
        self.assertEqual(repr(x), '[2, 3, 2, 3]')

    def test_checked_list_index(self):
        if False:
            print('Hello World!')
        x = CheckedList[int]()
        x.extend([1, 2, 3, 1, 2, 3])
        self.assertEqual(x.index(1), 0)
        x.remove(1)
        self.assertEqual(x.index(1), 2)
        x.remove(1)
        with self.assertRaises(ValueError):
            x.index(1)

    def test_checked_list_count(self):
        if False:
            print('Hello World!')
        x = CheckedList[int]()
        x.extend([1, 2, 3, 1, 2, 3])
        self.assertEqual(x.count(1), 2)
        x.remove(1)
        self.assertEqual(x.count(1), 1)

    def test_checked_list_reverse(self):
        if False:
            while True:
                i = 10
        x = CheckedList[int]()
        x.extend([1, 2, 3, 4])
        self.assertEqual(repr(x), '[1, 2, 3, 4]')
        self.assertEqual(x.reverse(), None)
        self.assertEqual(repr(type(x)), "<class '__static__.chklist[int]'>")
        self.assertEqual(repr(x), '[4, 3, 2, 1]')

    def test_checked_list_sort(self):
        if False:
            while True:
                i = 10
        x = CheckedList[int]()
        x.extend([3, 1, 2, 4])
        self.assertEqual(x.sort(), None)
        self.assertEqual(repr(type(x)), "<class '__static__.chklist[int]'>")
        self.assertEqual(repr(x), '[1, 2, 3, 4]')

    def test_checked_list_richcompare(self):
        if False:
            print('Hello World!')
        x = CheckedList[int]()
        x.extend([1, 2, 3])
        self.assertEqual(x < x, False)
        self.assertEqual(x <= x, True)
        y = CheckedList[int]()
        y.extend([2, 2, 3])
        self.assertEqual(y <= x, False)
        self.assertEqual(x < y, True)
        z = CheckedList[str]()
        z.extend(['a', 'a'])
        with self.assertRaises(TypeError):
            x < z
        self.assertEqual(x < [2, 2, 3], True)
        self.assertEqual([1, 2, 2] < x, True)
        self.assertEqual(x == [1, 2, 3], True)

    def test_checked_list_assign_subscript(self):
        if False:
            return 10
        x = CheckedList[int]([1, 2, 3, 4])
        self.assertEqual(x[2], 3)
        x[2] = 2
        self.assertEqual(x[2], 2)
        with self.assertRaises(TypeError):
            x[2] = 'A'
        self.assertEqual(x[2], 2)

    def test_checked_list_init(self):
        if False:
            while True:
                i = 10
        x = CheckedList[int]()
        self.assertEqual(repr(x), '[]')
        x = CheckedList[int]([1, 2, 3])
        self.assertEqual(repr(x), '[1, 2, 3]')
        self.assertEqual(repr(type(x)), "<class '__static__.chklist[int]'>")
        with self.assertRaises(TypeError):
            CheckedList[str]([1, 2, 3])
        x = CheckedList[int]({3: 'a', 2: 'b', 1: 'c'})
        self.assertEqual(repr(x), '[3, 2, 1]')
        with self.assertRaisesRegex(TypeError, 'chklist\\(\\) takes no keyword arguments'):
            CheckedList[int](iterable=[1, 2, 3])
        with self.assertRaisesRegex(TypeError, 'chklist expected at most 1 argument, got 2'):
            CheckedList[int]([], [])

    def test_checked_list_getitem_bad_return_type(self):
        if False:
            return 10
        codestr = '\n            from __static__ import CheckedList\n            def testfunc(x: CheckedList[int]) -> CheckedList[int]:\n                return x[1]\n        '
        self.type_error(codestr, bad_ret_type('int', 'chklist[int]'))

    def test_checked_list_getitem_slice_bad_return_type(self):
        if False:
            return 10
        codestr = '\n            from __static__ import CheckedList\n            def testfunc(x: CheckedList[int]) -> int:\n                return x[1:2]\n        '
        self.type_error(codestr, bad_ret_type('chklist[int]', 'int'))

    def test_checked_list_compile_getitem(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__ import CheckedList\n            def testfunc(x: CheckedList[int]) -> int:\n                return x[1]\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            l = CheckedList[int]([1, 2, 3])
            self.assertEqual(f(l), 2)
        codestr = '\n            from __static__ import CheckedList\n            def testfunc(x: CheckedList[int]) -> CheckedList[int]:\n                return x[1:2]\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            l = CheckedList[int]([1, 2, 3])
            self.assertEqual(f(l), [2])

    def test_checked_list_compile_setitem(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__ import CheckedList\n            def assign_to_index_1(x: CheckedList[int]) -> None:\n                x[1] = 2\n        '
        with self.in_module(codestr) as mod:
            f = mod.assign_to_index_1
            l = CheckedList[int]([1, 1, 1])
            self.assertEqual(f(l), None)
            self.assertEqual(repr(l), '[1, 2, 1]')

    def test_checked_list_compile_setitem_bad_type(self):
        if False:
            return 10
        codestr = '\n            from __static__ import CheckedList\n            def assign_to_index_1(x: CheckedList[int]) -> None:\n                x[1] = "a"\n        '
        self.type_error(codestr, type_mismatch('str', 'int'))

    def test_checked_list_compile_setitem_slice(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from __static__ import CheckedList\n            def assign_to_slice(x: CheckedList[int]) -> None:\n                x[1:3] = CheckedList[int]([2, 3])\n         '
        with self.in_module(codestr) as mod:
            f = mod.assign_to_slice
            l = CheckedList[int]([1, 1, 1])
            self.assertEqual(f(l), None)
            self.assertEqual(repr(l), '[1, 2, 3]')

    def test_checked_list_compile_setitem_slice_list(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__ import CheckedList\n            def assign_to_slice(x: CheckedList[int]) -> None:\n                x[1:3] = [2, 3]\n         '
        with self.in_module(codestr) as mod:
            f = mod.assign_to_slice
            l = CheckedList[int]([1, 1, 1])
            self.assertEqual(f(l), None)
            self.assertEqual(repr(l), '[1, 2, 3]')

    def test_checked_list_compile_setitem_slice_list_bad_index_type(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__ import CheckedList\n            def assign_to_slice(x: CheckedList[int]) -> None:\n                x["A"] = [2, 3]\n         '
        self.type_error(codestr, type_mismatch('str', 'int'))

    def test_checked_list_compile_jumpif(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            from __static__ import CheckedList\n            def testfunc(x: CheckedList[int]) -> int:\n                if x:\n                    return 1\n                return 2\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            l = CheckedList[int]([])
            self.assertEqual(f(l), 2)
            l.append(1)
            self.assertEqual(f(l), 1)

    def test_checked_list_compile_len(self):
        if False:
            return 10
        codestr = '\n            from __static__ import CheckedList\n            def testfunc(x: CheckedList[int]) -> int:\n                return len(x)\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            l = CheckedList[int]([])
            self.assertEqual(f(l), 0)
            l.append(1)
            self.assertEqual(f(l), 1)

    def test_checked_list_getitem_with_c_ints(self):
        if False:
            return 10
        codestr = '\n            from __static__ import CheckedList, int64, unbox\n            from typing import List\n            def testfunc(x: CheckedList[int]) -> int:\n                i: int64 = 0\n                j: int64 = 1\n                return x[i] + x[j]\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            cl = CheckedList[int]([1, 42, 3, 4, 5, 6])
            self.assertInBytecode(f, 'SEQUENCE_GET', SEQ_CHECKED_LIST)
            self.assertEqual(f(cl), 43)

    def test_checked_list_literal_basic(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from __static__ import CheckedList\n            def testfunc():\n                a: CheckedList[int] = [1, 2, 3, 4]\n                return a\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            l = f()
            self.assertInBytecode(f, 'BUILD_CHECKED_LIST')
            self.assertEqual(repr(l), '[1, 2, 3, 4]')
            self.assertEqual(type(l), CheckedList[int])

    def test_checked_list_literal_type_error(self):
        if False:
            return 10
        codestr = '\n            from __static__ import CheckedList\n            def testfunc():\n                a: CheckedList[int] = [1, 2, 3, "a"]\n                return a\n        '
        self.type_error(codestr, type_mismatch('chklist[Union[int, str]]', 'chklist[int]'))

    def test_checked_list_literal_basic_unpack(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__ import CheckedList\n            def testfunc():\n                a: CheckedList[int] = [1, 2, 3, 4]\n                b: CheckedList[int] = [*a]\n                return b\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            l = f()
            self.assert_jitted(f)
            self.assertInBytecode(f, 'BUILD_CHECKED_LIST')
            self.assertEqual(repr(l), '[1, 2, 3, 4]')
            self.assertEqual(type(l), CheckedList[int])

    def test_checked_list_literal_unpack_with_elements(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            from __static__ import CheckedList\n            def testfunc():\n                a: CheckedList[int] = [1, 2, 3, 4]\n                b: CheckedList[int] = [5, *a]\n                return b\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            l = f()
            self.assertInBytecode(f, 'BUILD_CHECKED_LIST')
            self.assertEqual(repr(l), '[5, 1, 2, 3, 4]')
            self.assertEqual(type(l), CheckedList[int])

    def test_checked_list_literal_comprehension(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__ import CheckedList\n            def testfunc():\n                a: CheckedList[int] = [int(x + 1) for x in [1, 2, 3, 4]]\n                return a\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            l = f()
            self.assertEqual(repr(l), '[2, 3, 4, 5]')
            self.assertEqual(type(l), CheckedList[int])

    def test_checked_list_literal_comprehension_type_error(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            from __static__ import CheckedList\n            def testfunc():\n                a: CheckedList[int] = [str(x + 1) for x in [1, 2, 3, 4]]\n                return a\n        '
        self.type_error(codestr, type_mismatch('chklist[str]', 'chklist[int]'))

    def test_checked_list_literal_opt_in(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__.compiler_flags import checked_lists\n\n            def testfunc():\n                a = [1, 2, 3, 4]\n                return a\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            l = f()
            self.assertInBytecode(f, 'BUILD_CHECKED_LIST')
            self.assertEqual(repr(l), '[1, 2, 3, 4]')
            self.assertEqual(type(l), CheckedList[int])

    def test_checked_list_call_opt_in(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__.compiler_flags import checked_lists\n\n            def testfunc():\n                a = list[int]((1, 2))\n                return a\n        '
        with self.in_module(codestr) as mod:
            l = mod.testfunc()
            self.assertEqual(repr(l), '[1, 2]')
            self.assertEqual(type(l), CheckedList[int])

    def test_checked_list_getitem_error(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__ import CheckedList, int64\n\n            def testfunc():\n                a: CheckedList[int] = [int(x + 1) for x in [1, 2, 3, 4]]\n                return a[int64(25)]\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            self.assertInBytecode(f, 'SEQUENCE_GET', SEQ_CHECKED_LIST)
            with self.assertRaises(IndexError):
                self.assertEqual(f(), 3)

    def test_checked_list_literal_opt_in_with_explicit_list_annotation_type_error(self):
        if False:
            return 10
        codestr = '\n            from __static__.compiler_flags import checked_lists\n            from typing import List\n\n            def testfunc():\n                a: List[int] = [1, 2, 3, 4]\n                return a\n        '
        self.type_error(codestr, type_mismatch('chklist[int]', 'list'))

    def test_checked_list_iter(self) -> None:
        if False:
            return 10
        codestr = '\n            from __static__ import CheckedList\n\n            def testfunc() -> int:\n                a: CheckedList[int] = [0, 1, 2]\n                for x in a:\n                    reveal_type(x)\n                return 0\n        '
        self.type_error(codestr, "reveal_type\\(x\\): 'int'", at='reveal_type')

    def test_fast_forloop(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__ import CheckedList\n\n            def f() -> int:\n                a: CheckedList[int] = [1, 2, 3]\n                sum: int = 0\n                for el in a:\n                    sum += el\n                return sum\n        '
        with self.in_module(codestr) as mod:
            self.assertInBytecode(mod.f, 'SEQUENCE_GET', SEQ_CHECKED_LIST | SEQ_SUBSCR_UNCHECKED)
            self.assertEqual(mod.f(), 6)

    def test_build_checked_list_cached(self):
        if False:
            print('Hello World!')
        codestr = '\n            from __static__ import CheckedList\n\n            def f() -> int:\n                a: CheckedList[int] = [1, 2, 3]\n                return sum(a)\n        '
        with self.in_module(codestr) as mod:
            self.assertInBytecode(mod.f, 'BUILD_CHECKED_LIST')
            for i in range(50):
                mod.f()
            self.assertEqual(mod.f(), 6)
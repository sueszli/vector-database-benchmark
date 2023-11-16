"""
Tests common to list and UserList.UserList
"""
import sys
import os
from functools import cmp_to_key
from test import support, seq_tests
from test.support import ALWAYS_EQ, NEVER_EQ

class CommonTest(seq_tests.CommonTest):

    def test_init(self):
        if False:
            return 10
        self.assertEqual(self.type2test([]), self.type2test())
        a = self.type2test([1, 2, 3])
        a.__init__()
        self.assertEqual(a, self.type2test([]))
        a = self.type2test([1, 2, 3])
        a.__init__([4, 5, 6])
        self.assertEqual(a, self.type2test([4, 5, 6]))
        b = self.type2test(a)
        self.assertNotEqual(id(a), id(b))
        self.assertEqual(a, b)

    def test_getitem_error(self):
        if False:
            print('Hello World!')
        a = []
        msg = 'list indices must be integers or slices'
        with self.assertRaisesRegex(TypeError, msg):
            a['a']

    def test_setitem_error(self):
        if False:
            i = 10
            return i + 15
        a = []
        msg = 'list indices must be integers or slices'
        with self.assertRaisesRegex(TypeError, msg):
            a['a'] = 'python'

    def test_repr(self):
        if False:
            return 10
        l0 = []
        l2 = [0, 1, 2]
        a0 = self.type2test(l0)
        a2 = self.type2test(l2)
        self.assertEqual(str(a0), str(l0))
        self.assertEqual(repr(a0), repr(l0))
        self.assertEqual(repr(a2), repr(l2))
        self.assertEqual(str(a2), '[0, 1, 2]')
        self.assertEqual(repr(a2), '[0, 1, 2]')
        a2.append(a2)
        a2.append(3)
        self.assertEqual(str(a2), '[0, 1, 2, [...], 3]')
        self.assertEqual(repr(a2), '[0, 1, 2, [...], 3]')

    def test_repr_deep(self):
        if False:
            return 10
        a = self.type2test([])
        for i in range(sys.getrecursionlimit() + 100):
            a = self.type2test([a])
        self.assertRaises(RecursionError, repr, a)

    def test_set_subscript(self):
        if False:
            i = 10
            return i + 15
        a = self.type2test(range(20))
        self.assertRaises(ValueError, a.__setitem__, slice(0, 10, 0), [1, 2, 3])
        self.assertRaises(TypeError, a.__setitem__, slice(0, 10), 1)
        self.assertRaises(ValueError, a.__setitem__, slice(0, 10, 2), [1, 2])
        self.assertRaises(TypeError, a.__getitem__, 'x', 1)
        a[slice(2, 10, 3)] = [1, 2, 3]
        self.assertEqual(a, self.type2test([0, 1, 1, 3, 4, 2, 6, 7, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]))

    def test_reversed(self):
        if False:
            i = 10
            return i + 15
        a = self.type2test(range(20))
        r = reversed(a)
        self.assertEqual(list(r), self.type2test(range(19, -1, -1)))
        self.assertRaises(StopIteration, next, r)
        self.assertEqual(list(reversed(self.type2test())), self.type2test())
        self.assertRaises(TypeError, len, reversed([1, 2, 3]))

    def test_setitem(self):
        if False:
            print('Hello World!')
        a = self.type2test([0, 1])
        a[0] = 0
        a[1] = 100
        self.assertEqual(a, self.type2test([0, 100]))
        a[-1] = 200
        self.assertEqual(a, self.type2test([0, 200]))
        a[-2] = 100
        self.assertEqual(a, self.type2test([100, 200]))
        self.assertRaises(IndexError, a.__setitem__, -3, 200)
        self.assertRaises(IndexError, a.__setitem__, 2, 200)
        a = self.type2test([])
        self.assertRaises(IndexError, a.__setitem__, 0, 200)
        self.assertRaises(IndexError, a.__setitem__, -1, 200)
        self.assertRaises(TypeError, a.__setitem__)
        a = self.type2test([0, 1, 2, 3, 4])
        a[0] = 1
        a[1] = 2
        a[2] = 3
        self.assertEqual(a, self.type2test([1, 2, 3, 3, 4]))
        a[0] = 5
        a[1] = 6
        a[2] = 7
        self.assertEqual(a, self.type2test([5, 6, 7, 3, 4]))
        a[-2] = 88
        a[-1] = 99
        self.assertEqual(a, self.type2test([5, 6, 7, 88, 99]))
        a[-2] = 8
        a[-1] = 9
        self.assertEqual(a, self.type2test([5, 6, 7, 8, 9]))
        msg = 'list indices must be integers or slices'
        with self.assertRaisesRegex(TypeError, msg):
            a['a'] = 'python'

    def test_delitem(self):
        if False:
            print('Hello World!')
        a = self.type2test([0, 1])
        del a[1]
        self.assertEqual(a, [0])
        del a[0]
        self.assertEqual(a, [])
        a = self.type2test([0, 1])
        del a[-2]
        self.assertEqual(a, [1])
        del a[-1]
        self.assertEqual(a, [])
        a = self.type2test([0, 1])
        self.assertRaises(IndexError, a.__delitem__, -3)
        self.assertRaises(IndexError, a.__delitem__, 2)
        a = self.type2test([])
        self.assertRaises(IndexError, a.__delitem__, 0)
        self.assertRaises(TypeError, a.__delitem__)

    def test_setslice(self):
        if False:
            for i in range(10):
                print('nop')
        l = [0, 1]
        a = self.type2test(l)
        for i in range(-3, 4):
            a[:i] = l[:i]
            self.assertEqual(a, l)
            a2 = a[:]
            a2[:i] = a[:i]
            self.assertEqual(a2, a)
            a[i:] = l[i:]
            self.assertEqual(a, l)
            a2 = a[:]
            a2[i:] = a[i:]
            self.assertEqual(a2, a)
            for j in range(-3, 4):
                a[i:j] = l[i:j]
                self.assertEqual(a, l)
                a2 = a[:]
                a2[i:j] = a[i:j]
                self.assertEqual(a2, a)
        aa2 = a2[:]
        aa2[:0] = [-2, -1]
        self.assertEqual(aa2, [-2, -1, 0, 1])
        aa2[0:] = []
        self.assertEqual(aa2, [])
        a = self.type2test([1, 2, 3, 4, 5])
        a[:-1] = a
        self.assertEqual(a, self.type2test([1, 2, 3, 4, 5, 5]))
        a = self.type2test([1, 2, 3, 4, 5])
        a[1:] = a
        self.assertEqual(a, self.type2test([1, 1, 2, 3, 4, 5]))
        a = self.type2test([1, 2, 3, 4, 5])
        a[1:-1] = a
        self.assertEqual(a, self.type2test([1, 1, 2, 3, 4, 5, 5]))
        a = self.type2test([])
        a[:] = tuple(range(10))
        self.assertEqual(a, self.type2test(range(10)))
        self.assertRaises(TypeError, a.__setitem__, slice(0, 1, 5))
        self.assertRaises(TypeError, a.__setitem__)

    def test_delslice(self):
        if False:
            for i in range(10):
                print('nop')
        a = self.type2test([0, 1])
        del a[1:2]
        del a[0:1]
        self.assertEqual(a, self.type2test([]))
        a = self.type2test([0, 1])
        del a[1:2]
        del a[0:1]
        self.assertEqual(a, self.type2test([]))
        a = self.type2test([0, 1])
        del a[-2:-1]
        self.assertEqual(a, self.type2test([1]))
        a = self.type2test([0, 1])
        del a[-2:-1]
        self.assertEqual(a, self.type2test([1]))
        a = self.type2test([0, 1])
        del a[1:]
        del a[:1]
        self.assertEqual(a, self.type2test([]))
        a = self.type2test([0, 1])
        del a[1:]
        del a[:1]
        self.assertEqual(a, self.type2test([]))
        a = self.type2test([0, 1])
        del a[-1:]
        self.assertEqual(a, self.type2test([0]))
        a = self.type2test([0, 1])
        del a[-1:]
        self.assertEqual(a, self.type2test([0]))
        a = self.type2test([0, 1])
        del a[:]
        self.assertEqual(a, self.type2test([]))

    def test_append(self):
        if False:
            while True:
                i = 10
        a = self.type2test([])
        a.append(0)
        a.append(1)
        a.append(2)
        self.assertEqual(a, self.type2test([0, 1, 2]))
        self.assertRaises(TypeError, a.append)

    def test_extend(self):
        if False:
            i = 10
            return i + 15
        a1 = self.type2test([0])
        a2 = self.type2test((0, 1))
        a = a1[:]
        a.extend(a2)
        self.assertEqual(a, a1 + a2)
        a.extend(self.type2test([]))
        self.assertEqual(a, a1 + a2)
        a.extend(a)
        self.assertEqual(a, self.type2test([0, 0, 1, 0, 0, 1]))
        a = self.type2test('spam')
        a.extend('eggs')
        self.assertEqual(a, list('spameggs'))
        self.assertRaises(TypeError, a.extend, None)
        self.assertRaises(TypeError, a.extend)

        class CustomIter:

            def __iter__(self):
                if False:
                    return 10
                return self

            def __next__(self):
                if False:
                    print('Hello World!')
                raise StopIteration

            def __length_hint__(self):
                if False:
                    i = 10
                    return i + 15
                return sys.maxsize
        a = self.type2test([1, 2, 3, 4])
        a.extend(CustomIter())
        self.assertEqual(a, [1, 2, 3, 4])

    def test_insert(self):
        if False:
            i = 10
            return i + 15
        a = self.type2test([0, 1, 2])
        a.insert(0, -2)
        a.insert(1, -1)
        a.insert(2, 0)
        self.assertEqual(a, [-2, -1, 0, 0, 1, 2])
        b = a[:]
        b.insert(-2, 'foo')
        b.insert(-200, 'left')
        b.insert(200, 'right')
        self.assertEqual(b, self.type2test(['left', -2, -1, 0, 0, 'foo', 1, 2, 'right']))
        self.assertRaises(TypeError, a.insert)

    def test_pop(self):
        if False:
            i = 10
            return i + 15
        a = self.type2test([-1, 0, 1])
        a.pop()
        self.assertEqual(a, [-1, 0])
        a.pop(0)
        self.assertEqual(a, [0])
        self.assertRaises(IndexError, a.pop, 5)
        a.pop(0)
        self.assertEqual(a, [])
        self.assertRaises(IndexError, a.pop)
        self.assertRaises(TypeError, a.pop, 42, 42)
        a = self.type2test([0, 10, 20, 30, 40])

    def test_remove(self):
        if False:
            while True:
                i = 10
        a = self.type2test([0, 0, 1])
        a.remove(1)
        self.assertEqual(a, [0, 0])
        a.remove(0)
        self.assertEqual(a, [0])
        a.remove(0)
        self.assertEqual(a, [])
        self.assertRaises(ValueError, a.remove, 0)
        self.assertRaises(TypeError, a.remove)
        a = self.type2test([1, 2])
        self.assertRaises(ValueError, a.remove, NEVER_EQ)
        self.assertEqual(a, [1, 2])
        a.remove(ALWAYS_EQ)
        self.assertEqual(a, [2])
        a = self.type2test([ALWAYS_EQ])
        a.remove(1)
        self.assertEqual(a, [])
        a = self.type2test([ALWAYS_EQ])
        a.remove(NEVER_EQ)
        self.assertEqual(a, [])
        a = self.type2test([NEVER_EQ])
        self.assertRaises(ValueError, a.remove, ALWAYS_EQ)

        class BadExc(Exception):
            pass

        class BadCmp:

            def __eq__(self, other):
                if False:
                    print('Hello World!')
                if other == 2:
                    raise BadExc()
                return False
        a = self.type2test([0, 1, 2, 3])
        self.assertRaises(BadExc, a.remove, BadCmp())

        class BadCmp2:

            def __eq__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                raise BadExc()
        d = self.type2test('abcdefghcij')
        d.remove('c')
        self.assertEqual(d, self.type2test('abdefghcij'))
        d.remove('c')
        self.assertEqual(d, self.type2test('abdefghij'))
        self.assertRaises(ValueError, d.remove, 'c')
        self.assertEqual(d, self.type2test('abdefghij'))
        d = self.type2test(['a', 'b', BadCmp2(), 'c'])
        e = self.type2test(d)
        self.assertRaises(BadExc, d.remove, 'c')
        for (x, y) in zip(d, e):
            self.assertIs(x, y)

    def test_index(self):
        if False:
            print('Hello World!')
        super().test_index()
        a = self.type2test([-2, -1, 0, 0, 1, 2])
        a.remove(0)
        self.assertRaises(ValueError, a.index, 2, 0, 4)
        self.assertEqual(a, self.type2test([-2, -1, 0, 1, 2]))

        class EvilCmp:

            def __init__(self, victim):
                if False:
                    i = 10
                    return i + 15
                self.victim = victim

            def __eq__(self, other):
                if False:
                    while True:
                        i = 10
                del self.victim[:]
                return False
        a = self.type2test()
        a[:] = [EvilCmp(a) for _ in range(100)]
        self.assertRaises(ValueError, a.index, None)

    def test_reverse(self):
        if False:
            while True:
                i = 10
        u = self.type2test([-2, -1, 0, 1, 2])
        u2 = u[:]
        u.reverse()
        self.assertEqual(u, [2, 1, 0, -1, -2])
        u.reverse()
        self.assertEqual(u, u2)
        self.assertRaises(TypeError, u.reverse, 42)

    def test_clear(self):
        if False:
            return 10
        u = self.type2test([2, 3, 4])
        u.clear()
        self.assertEqual(u, [])
        u = self.type2test([])
        u.clear()
        self.assertEqual(u, [])
        u = self.type2test([])
        u.append(1)
        u.clear()
        u.append(2)
        self.assertEqual(u, [2])
        self.assertRaises(TypeError, u.clear, None)

    def test_copy(self):
        if False:
            while True:
                i = 10
        u = self.type2test([1, 2, 3])
        v = u.copy()
        self.assertEqual(v, [1, 2, 3])
        u = self.type2test([])
        v = u.copy()
        self.assertEqual(v, [])
        u = self.type2test(['a', 'b'])
        v = u.copy()
        v.append('i')
        self.assertEqual(u, ['a', 'b'])
        self.assertEqual(v, u + ['i'])
        u = self.type2test([1, 2, [3, 4], 5])
        v = u.copy()
        self.assertEqual(u, v)
        self.assertIs(v[3], u[3])
        self.assertRaises(TypeError, u.copy, None)

    def test_sort(self):
        if False:
            print('Hello World!')
        u = self.type2test([1, 0])
        u.sort()
        self.assertEqual(u, [0, 1])
        u = self.type2test([2, 1, 0, -1, -2])
        u.sort()
        self.assertEqual(u, self.type2test([-2, -1, 0, 1, 2]))
        self.assertRaises(TypeError, u.sort, 42, 42)

        def revcmp(a, b):
            if False:
                while True:
                    i = 10
            if a == b:
                return 0
            elif a < b:
                return 1
            else:
                return -1
        u.sort(key=cmp_to_key(revcmp))
        self.assertEqual(u, self.type2test([2, 1, 0, -1, -2]))

        def myComparison(x, y):
            if False:
                return 10
            (xmod, ymod) = (x % 3, y % 7)
            if xmod == ymod:
                return 0
            elif xmod < ymod:
                return -1
            else:
                return 1
        z = self.type2test(range(12))
        z.sort(key=cmp_to_key(myComparison))
        self.assertRaises(TypeError, z.sort, 2)

        def selfmodifyingComparison(x, y):
            if False:
                while True:
                    i = 10
            z.append(1)
            if x == y:
                return 0
            elif x < y:
                return -1
            else:
                return 1
        self.assertRaises(ValueError, z.sort, key=cmp_to_key(selfmodifyingComparison))
        self.assertRaises(TypeError, z.sort, 42, 42, 42, 42)

    def test_slice(self):
        if False:
            return 10
        u = self.type2test('spam')
        u[:2] = 'h'
        self.assertEqual(u, list('ham'))

    def test_iadd(self):
        if False:
            while True:
                i = 10
        super().test_iadd()
        u = self.type2test([0, 1])
        u2 = u
        u += [2, 3]
        self.assertIs(u, u2)
        u = self.type2test('spam')
        u += 'eggs'
        self.assertEqual(u, self.type2test('spameggs'))
        self.assertRaises(TypeError, u.__iadd__, None)

    def test_imul(self):
        if False:
            i = 10
            return i + 15
        super().test_imul()
        s = self.type2test([])
        oldid = id(s)
        s *= 10
        self.assertEqual(id(s), oldid)

    def test_extendedslicing(self):
        if False:
            while True:
                i = 10
        a = self.type2test([0, 1, 2, 3, 4])
        del a[::2]
        self.assertEqual(a, self.type2test([1, 3]))
        a = self.type2test(range(5))
        del a[1::2]
        self.assertEqual(a, self.type2test([0, 2, 4]))
        a = self.type2test(range(5))
        del a[1::-2]
        self.assertEqual(a, self.type2test([0, 2, 3, 4]))
        a = self.type2test(range(10))
        del a[::1000]
        self.assertEqual(a, self.type2test([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        a = self.type2test(range(10))
        a[::2] = [-1] * 5
        self.assertEqual(a, self.type2test([-1, 1, -1, 3, -1, 5, -1, 7, -1, 9]))
        a = self.type2test(range(10))
        a[::-4] = [10] * 3
        self.assertEqual(a, self.type2test([0, 10, 2, 3, 4, 10, 6, 7, 8, 10]))
        a = self.type2test(range(4))
        a[::-1] = a
        self.assertEqual(a, self.type2test([3, 2, 1, 0]))
        a = self.type2test(range(10))
        b = a[:]
        c = a[:]
        a[2:3] = self.type2test(['two', 'elements'])
        b[slice(2, 3)] = self.type2test(['two', 'elements'])
        c[2:3] = self.type2test(['two', 'elements'])
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        a = self.type2test(range(10))
        a[::2] = tuple(range(5))
        self.assertEqual(a, self.type2test([0, 1, 1, 3, 2, 5, 3, 7, 4, 9]))
        a = self.type2test(range(10))
        del a[9::1 << 333]

    def test_constructor_exception_handling(self):
        if False:
            for i in range(10):
                print('nop')

        class F(object):

            def __iter__(self):
                if False:
                    i = 10
                    return i + 15
                raise KeyboardInterrupt
        self.assertRaises(KeyboardInterrupt, list, F())

    def test_exhausted_iterator(self):
        if False:
            return 10
        a = self.type2test([1, 2, 3])
        exhit = iter(a)
        empit = iter(a)
        for x in exhit:
            next(empit)
        a.append(9)
        self.assertEqual(list(exhit), [])
        self.assertEqual(list(empit), [9])
        self.assertEqual(a, self.type2test([1, 2, 3, 9]))
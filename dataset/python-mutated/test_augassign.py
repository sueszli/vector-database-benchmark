import unittest

class AugAssignTest(unittest.TestCase):

    def testBasic(self):
        if False:
            i = 10
            return i + 15
        x = 2
        x += 1
        x *= 2
        x **= 2
        x -= 8
        x //= 5
        x %= 3
        x &= 2
        x |= 5
        x ^= 1
        x /= 2
        self.assertEqual(x, 3.0)

    def test_with_unpacking(self):
        if False:
            return 10
        self.assertRaises(SyntaxError, compile, 'x, b += 3', '<test>', 'exec')

    def testInList(self):
        if False:
            i = 10
            return i + 15
        x = [2]
        x[0] += 1
        x[0] *= 2
        x[0] **= 2
        x[0] -= 8
        x[0] //= 5
        x[0] %= 3
        x[0] &= 2
        x[0] |= 5
        x[0] ^= 1
        x[0] /= 2
        self.assertEqual(x[0], 3.0)

    def testInDict(self):
        if False:
            return 10
        x = {0: 2}
        x[0] += 1
        x[0] *= 2
        x[0] **= 2
        x[0] -= 8
        x[0] //= 5
        x[0] %= 3
        x[0] &= 2
        x[0] |= 5
        x[0] ^= 1
        x[0] /= 2
        self.assertEqual(x[0], 3.0)

    def testSequences(self):
        if False:
            while True:
                i = 10
        x = [1, 2]
        x += [3, 4]
        x *= 2
        self.assertEqual(x, [1, 2, 3, 4, 1, 2, 3, 4])
        x = [1, 2, 3]
        y = x
        x[1:2] *= 2
        y[1:2] += [1]
        self.assertEqual(x, [1, 2, 1, 2, 3])
        self.assertTrue(x is y)

    def testCustomMethods1(self):
        if False:
            while True:
                i = 10

        class aug_test:

            def __init__(self, value):
                if False:
                    for i in range(10):
                        print('nop')
                self.val = value

            def __radd__(self, val):
                if False:
                    print('Hello World!')
                return self.val + val

            def __add__(self, val):
                if False:
                    i = 10
                    return i + 15
                return aug_test(self.val + val)

        class aug_test2(aug_test):

            def __iadd__(self, val):
                if False:
                    print('Hello World!')
                self.val = self.val + val
                return self

        class aug_test3(aug_test):

            def __iadd__(self, val):
                if False:
                    while True:
                        i = 10
                return aug_test3(self.val + val)

        class aug_test4(aug_test3):
            """Blocks inheritance, and fallback to __add__"""
            __iadd__ = None
        x = aug_test(1)
        y = x
        x += 10
        self.assertIsInstance(x, aug_test)
        self.assertTrue(y is not x)
        self.assertEqual(x.val, 11)
        x = aug_test2(2)
        y = x
        x += 10
        self.assertTrue(y is x)
        self.assertEqual(x.val, 12)
        x = aug_test3(3)
        y = x
        x += 10
        self.assertIsInstance(x, aug_test3)
        self.assertTrue(y is not x)
        self.assertEqual(x.val, 13)
        x = aug_test4(4)
        with self.assertRaises(TypeError):
            x += 10

    def testCustomMethods2(test_self):
        if False:
            while True:
                i = 10
        output = []

        class testall:

            def __add__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__add__ called')

            def __radd__(self, val):
                if False:
                    print('Hello World!')
                output.append('__radd__ called')

            def __iadd__(self, val):
                if False:
                    print('Hello World!')
                output.append('__iadd__ called')
                return self

            def __sub__(self, val):
                if False:
                    return 10
                output.append('__sub__ called')

            def __rsub__(self, val):
                if False:
                    print('Hello World!')
                output.append('__rsub__ called')

            def __isub__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__isub__ called')
                return self

            def __mul__(self, val):
                if False:
                    print('Hello World!')
                output.append('__mul__ called')

            def __rmul__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__rmul__ called')

            def __imul__(self, val):
                if False:
                    while True:
                        i = 10
                output.append('__imul__ called')
                return self

            def __matmul__(self, val):
                if False:
                    return 10
                output.append('__matmul__ called')

            def __rmatmul__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__rmatmul__ called')

            def __imatmul__(self, val):
                if False:
                    print('Hello World!')
                output.append('__imatmul__ called')
                return self

            def __floordiv__(self, val):
                if False:
                    while True:
                        i = 10
                output.append('__floordiv__ called')
                return self

            def __ifloordiv__(self, val):
                if False:
                    for i in range(10):
                        print('nop')
                output.append('__ifloordiv__ called')
                return self

            def __rfloordiv__(self, val):
                if False:
                    while True:
                        i = 10
                output.append('__rfloordiv__ called')
                return self

            def __truediv__(self, val):
                if False:
                    print('Hello World!')
                output.append('__truediv__ called')
                return self

            def __rtruediv__(self, val):
                if False:
                    for i in range(10):
                        print('nop')
                output.append('__rtruediv__ called')
                return self

            def __itruediv__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__itruediv__ called')
                return self

            def __mod__(self, val):
                if False:
                    print('Hello World!')
                output.append('__mod__ called')

            def __rmod__(self, val):
                if False:
                    print('Hello World!')
                output.append('__rmod__ called')

            def __imod__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__imod__ called')
                return self

            def __pow__(self, val):
                if False:
                    for i in range(10):
                        print('nop')
                output.append('__pow__ called')

            def __rpow__(self, val):
                if False:
                    print('Hello World!')
                output.append('__rpow__ called')

            def __ipow__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__ipow__ called')
                return self

            def __or__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__or__ called')

            def __ror__(self, val):
                if False:
                    while True:
                        i = 10
                output.append('__ror__ called')

            def __ior__(self, val):
                if False:
                    for i in range(10):
                        print('nop')
                output.append('__ior__ called')
                return self

            def __and__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__and__ called')

            def __rand__(self, val):
                if False:
                    print('Hello World!')
                output.append('__rand__ called')

            def __iand__(self, val):
                if False:
                    return 10
                output.append('__iand__ called')
                return self

            def __xor__(self, val):
                if False:
                    while True:
                        i = 10
                output.append('__xor__ called')

            def __rxor__(self, val):
                if False:
                    while True:
                        i = 10
                output.append('__rxor__ called')

            def __ixor__(self, val):
                if False:
                    print('Hello World!')
                output.append('__ixor__ called')
                return self

            def __rshift__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__rshift__ called')

            def __rrshift__(self, val):
                if False:
                    for i in range(10):
                        print('nop')
                output.append('__rrshift__ called')

            def __irshift__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__irshift__ called')
                return self

            def __lshift__(self, val):
                if False:
                    i = 10
                    return i + 15
                output.append('__lshift__ called')

            def __rlshift__(self, val):
                if False:
                    for i in range(10):
                        print('nop')
                output.append('__rlshift__ called')

            def __ilshift__(self, val):
                if False:
                    return 10
                output.append('__ilshift__ called')
                return self
        x = testall()
        x + 1
        1 + x
        x += 1
        x - 1
        1 - x
        x -= 1
        x * 1
        1 * x
        x *= 1
        x @ 1
        1 @ x
        x @= 1
        x / 1
        1 / x
        x /= 1
        x // 1
        1 // x
        x //= 1
        x % 1
        1 % x
        x %= 1
        x ** 1
        1 ** x
        x **= 1
        x | 1
        1 | x
        x |= 1
        x & 1
        1 & x
        x &= 1
        x ^ 1
        1 ^ x
        x ^= 1
        x >> 1
        1 >> x
        x >>= 1
        x << 1
        1 << x
        x <<= 1
        test_self.assertEqual(output, '__add__ called\n__radd__ called\n__iadd__ called\n__sub__ called\n__rsub__ called\n__isub__ called\n__mul__ called\n__rmul__ called\n__imul__ called\n__matmul__ called\n__rmatmul__ called\n__imatmul__ called\n__truediv__ called\n__rtruediv__ called\n__itruediv__ called\n__floordiv__ called\n__rfloordiv__ called\n__ifloordiv__ called\n__mod__ called\n__rmod__ called\n__imod__ called\n__pow__ called\n__rpow__ called\n__ipow__ called\n__or__ called\n__ror__ called\n__ior__ called\n__and__ called\n__rand__ called\n__iand__ called\n__xor__ called\n__rxor__ called\n__ixor__ called\n__rshift__ called\n__rrshift__ called\n__irshift__ called\n__lshift__ called\n__rlshift__ called\n__ilshift__ called\n'.splitlines())
if __name__ == '__main__':
    unittest.main()
import unittest
from numba.tests.support import captured_stdout

class DocsLiterallyUsageTest(unittest.TestCase):

    def test_literally_usage(self):
        if False:
            for i in range(10):
                print('nop')
        with captured_stdout() as stdout:
            import numba

            def power(x, n):
                if False:
                    i = 10
                    return i + 15
                raise NotImplementedError

            @numba.extending.overload(power)
            def ov_power(x, n):
                if False:
                    while True:
                        i = 10
                if isinstance(n, numba.types.Literal):
                    if n.literal_value == 2:
                        print('square')
                        return lambda x, n: x * x
                    elif n.literal_value == 3:
                        print('cubic')
                        return lambda x, n: x * x * x
                else:
                    return lambda x, n: numba.literally(n)
                print('generic')
                return lambda x, n: x ** n

            @numba.njit
            def test_power(x, n):
                if False:
                    print('Hello World!')
                return power(x, n)
            print(test_power(3, 2))
            print(test_power(3, 3))
            print(test_power(3, 4))
            assert test_power(3, 2) == 3 ** 2
            assert test_power(3, 3) == 3 ** 3
            assert test_power(3, 4) == 3 ** 4
        self.assertEqual('square\n9\ncubic\n27\ngeneric\n81\n', stdout.getvalue())
if __name__ == '__main__':
    unittest.main()
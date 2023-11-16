import unittest

class CFunctionCalls(unittest.TestCase):

    def test_varargs0(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, {}.__contains__)

    def test_varargs1(self):
        if False:
            for i in range(10):
                print('nop')
        {}.__contains__(0)

    def test_varargs2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, {}.__contains__, 0, 1)

    def test_varargs0_ext(self):
        if False:
            return 10
        try:
            {}.__contains__(*())
        except TypeError:
            pass

    def test_varargs1_ext(self):
        if False:
            return 10
        {}.__contains__(*(0,))

    def test_varargs2_ext(self):
        if False:
            return 10
        try:
            {}.__contains__(*(1, 2))
        except TypeError:
            pass
        else:
            raise RuntimeError

    def test_varargs0_kw(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, {}.__contains__, x=2)

    def test_varargs1_kw(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, {}.__contains__, x=2)

    def test_varargs2_kw(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, {}.__contains__, x=2, y=2)

    def test_oldargs0_0(self):
        if False:
            while True:
                i = 10
        {}.keys()

    def test_oldargs0_1(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, {}.keys, 0)

    def test_oldargs0_2(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, {}.keys, 0, 1)

    def test_oldargs0_0_ext(self):
        if False:
            while True:
                i = 10
        {}.keys(*())

    def test_oldargs0_1_ext(self):
        if False:
            while True:
                i = 10
        try:
            {}.keys(*(0,))
        except TypeError:
            pass
        else:
            raise RuntimeError

    def test_oldargs0_2_ext(self):
        if False:
            print('Hello World!')
        try:
            {}.keys(*(1, 2))
        except TypeError:
            pass
        else:
            raise RuntimeError

    def test_oldargs0_1_kw(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, {}.keys, x=2)

    def test_oldargs0_2_kw(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, {}.keys, x=2, y=2)

    def test_oldargs1_0(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, [].count)

    def test_oldargs1_1(self):
        if False:
            while True:
                i = 10
        [].count(1)

    def test_oldargs1_2(self):
        if False:
            return 10
        self.assertRaises(TypeError, [].count, 1, 2)

    def test_oldargs1_0_ext(self):
        if False:
            i = 10
            return i + 15
        try:
            [].count(*())
        except TypeError:
            pass
        else:
            raise RuntimeError

    def test_oldargs1_1_ext(self):
        if False:
            print('Hello World!')
        [].count(*(1,))

    def test_oldargs1_2_ext(self):
        if False:
            i = 10
            return i + 15
        try:
            [].count(*(1, 2))
        except TypeError:
            pass
        else:
            raise RuntimeError

    def test_oldargs1_0_kw(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, [].count, x=2)

    def test_oldargs1_1_kw(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, [].count, {}, x=2)

    def test_oldargs1_2_kw(self):
        if False:
            return 10
        self.assertRaises(TypeError, [].count, x=2, y=2)
if __name__ == '__main__':
    unittest.main()
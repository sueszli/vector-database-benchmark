import unittest
from paddle.jit.dy2static.utils import GetterSetterHelper
vars = [1, 2, 3, 4, 5]

def getter():
    if False:
        for i in range(10):
            print('nop')
    return vars

def setter(values):
    if False:
        for i in range(10):
            print('nop')
    global vars
    vars = values

class TestGetterSetterHelper(unittest.TestCase):

    def test_1(self):
        if False:
            return 10
        helper = GetterSetterHelper(getter, setter, ['a', 'b', 'e'], ['d', 'f', 'e'])
        print(helper.union())
        expect_union = ['a', 'b', 'd', 'e', 'f']
        assert helper.union() == expect_union
        assert helper.get(expect_union) == (1, 2, 3, 4, 5)
        helper.set(['a', 'b'], [1, 1])
        assert vars == [1, 1, 3, 4, 5]
        helper.set(['f', 'e'], [12, 10])
        assert vars == [1, 1, 3, 10, 12]
        helper.set(None, None)
        assert vars == [1, 1, 3, 10, 12]
        assert helper.get(None) == ()
        assert helper.get([]) == ()
if __name__ == '__main__':
    unittest.main()
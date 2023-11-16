import unittest
from typing import List, Union
from modelscope.utils.type_assert import type_assert

class type_assertTest(unittest.TestCase):

    @type_assert(object, list, (int, str))
    def a(self, a: List[int], b: Union[int, str]):
        if False:
            while True:
                i = 10
        print(a, b)

    def test_type_assert(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            self.a([1], 2)
            self.a(1, [123])
if __name__ == '__main__':
    unittest.main()
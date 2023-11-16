import unittest

class TestReverse(unittest.TestCase):

    def test_reverse(self, func):
        if False:
            return 10
        self.assertEqual(func(None), None)
        self.assertEqual(func(['']), [''])
        self.assertEqual(func(['f', 'o', 'o', ' ', 'b', 'a', 'r']), ['r', 'a', 'b', ' ', 'o', 'o', 'f'])
        print('Success: test_reverse')

    def test_reverse_inplace(self, func):
        if False:
            i = 10
            return i + 15
        target_list = ['f', 'o', 'o', ' ', 'b', 'a', 'r']
        func(target_list)
        self.assertEqual(target_list, ['r', 'a', 'b', ' ', 'o', 'o', 'f'])
        print('Success: test_reverse_inplace')

def main():
    if False:
        while True:
            i = 10
    test = TestReverse()
    reverse_string = ReverseString()
    test.test_reverse(reverse_string.reverse)
    test.test_reverse_inplace(reverse_string.reverse)
if __name__ == '__main__':
    main()
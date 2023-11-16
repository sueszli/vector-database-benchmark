import unittest

class TestSubTwo(unittest.TestCase):

    def test_sub_two(self):
        if False:
            return 10
        solution = Solution()
        self.assertRaises(TypeError, solution.sub_two, None)
        self.assertEqual(solution.sub_two(7, 5), 2)
        self.assertEqual(solution.sub_two(-5, -7), 2)
        self.assertEqual(solution.sub_two(-5, 7), -12)
        self.assertEqual(solution.sub_two(5, -7), 12)
        print('Success: test_sub_two')

def main():
    if False:
        i = 10
        return i + 15
    test = TestSubTwo()
    test.test_sub_two()
if __name__ == '__main__':
    main()
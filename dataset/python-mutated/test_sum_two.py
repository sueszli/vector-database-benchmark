import unittest

class TestSumTwo(unittest.TestCase):

    def test_sum_two(self):
        if False:
            i = 10
            return i + 15
        solution = Solution()
        self.assertRaises(TypeError, solution.sum_two, None)
        self.assertEqual(solution.sum_two(5, 7), 12)
        self.assertEqual(solution.sum_two(-5, -7), -12)
        self.assertEqual(solution.sum_two(5, -7), -2)
        print('Success: test_sum_two')

def main():
    if False:
        for i in range(10):
            print('nop')
    test = TestSumTwo()
    test.test_sum_two()
if __name__ == '__main__':
    main()
import unittest

class TestUtopianTree(unittest.TestCase):

    def test_utopian_tree(self):
        if False:
            while True:
                i = 10
        solution = Solution()
        self.assertEqual(solution.calc_utopian_tree_height(0), 1)
        self.assertEqual(solution.calc_utopian_tree_height(1), 2)
        self.assertEqual(solution.calc_utopian_tree_height(4), 7)
        print('Success: test_utopian_tree')

def main():
    if False:
        for i in range(10):
            print('nop')
    test = TestUtopianTree()
    test.test_utopian_tree()
if __name__ == '__main__':
    main()
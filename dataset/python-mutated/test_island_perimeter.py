import unittest

class TestIslandPerimeter(unittest.TestCase):

    def test_island_perimeter(self):
        if False:
            print('Hello World!')
        solution = Solution()
        self.assertRaises(TypeError, solution.island_perimeter, None)
        data = [[1, 0]]
        expected = 4
        self.assertEqual(solution.island_perimeter(data), expected)
        data = [[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]]
        expected = 16
        self.assertEqual(solution.island_perimeter(data), expected)
        print('Success: test_island_perimeter')

def main():
    if False:
        while True:
            i = 10
    test = TestIslandPerimeter()
    test.test_island_perimeter()
if __name__ == '__main__':
    main()
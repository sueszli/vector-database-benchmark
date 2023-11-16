import unittest

class TestMatrixMultiplicationCost(unittest.TestCase):

    def test_find_min_cost(self):
        if False:
            for i in range(10):
                print('nop')
        matrix_mult_cost = MatrixMultiplicationCost()
        self.assertRaises(TypeError, matrix_mult_cost.find_min_cost, None)
        self.assertEqual(matrix_mult_cost.find_min_cost([]), 0)
        matrices = [Matrix(2, 3), Matrix(3, 6), Matrix(6, 4), Matrix(4, 5)]
        expected_cost = 124
        self.assertEqual(matrix_mult_cost.find_min_cost(matrices), expected_cost)
        print('Success: test_find_min_cost')

def main():
    if False:
        i = 10
        return i + 15
    test = TestMatrixMultiplicationCost()
    test.test_find_min_cost()
if __name__ == '__main__':
    main()
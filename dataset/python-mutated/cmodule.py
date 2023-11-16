import vaex.dataset as dataset
import numpy as np
import unittest
import vaex as vx
import vaex.vaexfast

class TestStatisticNd(unittest.TestCase):

    def test_add(self):
        if False:
            return 10
        x = np.arange(10, dtype=np.float64)
        grid = np.zeros((10, 2), dtype=np.float64)
        w = x * 1
        w[2] = np.nan
        vaex.vaexfast.statisticNd([x], w, grid, [0.0], [10.0], 0)
        print(grid)
        grid0 = np.zeros((1,), dtype=np.float64)
        vaex.vaexfast.statisticNd([], w, grid0, [], [], 0)
        print(grid0)

    def test_2(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(10, dtype=np.float64) + 10
        grid = np.zeros(2, dtype=np.float64)
        grid[..., 0] = np.inf
        grid[..., 1] = -np.inf
        w = x * 1
        w[2] = np.nan
        print(np.nansum(w))
        vaex.vaexfast.statisticNd([], w, grid, [], [], 2)
        print(grid)

    def test_edges(self):
        if False:
            return 10
        grid = np.zeros((10, 1), dtype=np.float64)
        x = np.arange(10, dtype=np.float64)
        x[0] = np.nan
        vaex.vaexfast.statisticNd([x], None, grid, [4.0], [6.0], 0, False)
        print(grid.T)
        grid = np.zeros((10 + 3, 1), dtype=np.float64)
        x = np.arange(10, dtype=np.float64)
        x[0] = np.nan
        vaex.vaexfast.statisticNd([x], None, grid, [4.0], [6.0], 0, True)
        print(x)
        print(grid.T)
        self.assertEqual(sum(grid), len(x))
        self.assertEqual(grid[-1], 4)
        self.assertEqual(grid[1], 3)
        self.assertEqual(grid[0], 1)
        grid = np.zeros((10, 10, 1), dtype=np.float64)
        x = np.arange(10, dtype=np.float64)
        y = np.arange(10, dtype=np.float64)
        x[0] = np.nan
        y[-1] = np.nan
        y[-2] = np.nan
        x[1] = np.nan
        y[1] = np.nan
        vaex.vaexfast.statisticNd([x, y], None, grid, [4.0, 3.0], [6.0, 7.0], 0, True)
        print(grid[..., 0])
        print(grid.shape)
        self.assertEqual(np.sum(grid), len(x))
        self.assertEqual(grid[0, 0], 1)
        self.assertEqual(grid[0, 1], 1)
        self.assertEqual(grid[-1, 0], 2)

    def test_find_edges(self):
        if False:
            print('Hello World!')
        grid = np.zeros((10 + 3, 1), dtype=np.float64)
        x = np.arange(10, dtype=np.float64)
        vaex.vaexfast.statisticNd([x], None, grid, [0.0], [10.0], 0, True)
        print(grid.T)
        c = np.cumsum(grid[1:], axis=0)
        print(c.T, c.shape)
        c = c.reshape(-1)
        print(c, c.shape)
        values = np.array(4.5)
        print(values.T, values.shape)
        edges = np.zeros(2, dtype=np.int64)
        vaex.vaexfast.grid_find_edges(c, values, edges)
        print(edges)

class TestGridInterpolate(unittest.TestCase):

    def test_interpolate(self):
        if False:
            print('Hello World!')
        x = np.array([[0.0, 1.0]])
        y = np.array([2.0])
        vaex.vaexfast.grid_interpolate(x, y, 0.5)
        self.assertEqual(y[0], 0.5)
        x = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1]])
        vaex.vaexfast.grid_interpolate(x, y, 0.5)
        self.assertEqual(y[0], 1.0 / 4)
        x = np.array([[0, 0.5, 0.75, 1]])
        vaex.vaexfast.grid_interpolate(x, y, 0.5)
        self.assertEqual(y[0], 1.0 / 3)
        vaex.vaexfast.grid_interpolate(x, y, 0.75)
        self.assertEqual(y[0], 2.0 / 3)
        vaex.vaexfast.grid_interpolate(x, y, 0.5 + 0.25 / 2)
        self.assertEqual(y[0], 0.5)
        x = np.array([[0, 0.5, 0.75, 1], [0.5, 0.5, 0.75, 1], [0, 0.5, 1.0, 1]])
        y = np.array([2.0, 2.0, 2.0])
        vaex.vaexfast.grid_interpolate(x, y, 0.5)
        np.testing.assert_array_almost_equal(y, np.array([1 / 3.0, 1.0 / 3 / 2, 1.0 / 3.0]))
        x = np.array([[0, 0.5, 0.75, 1], [0.5, 0.5, 0.75, 1], [0, 0.5, 1.0, 1]])
        vaex.vaexfast.grid_interpolate(x, y, 0.75)
        np.testing.assert_array_almost_equal(y, np.array([2 / 3.0, 2.0 / 3, 1.0 / 2]))
        print('#######')
        x = np.array([[0, 0.5, 0.75, 1], [0.5, 0.5, 0.75, 1], [0, 0.5, 1.0, 1]])
        vaex.vaexfast.grid_interpolate(x, y, 1.0)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 1, 5.0 / 6]))
if __name__ == '__main__':
    unittest.main()
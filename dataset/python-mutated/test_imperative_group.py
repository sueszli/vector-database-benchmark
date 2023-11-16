import unittest
import paddle
from paddle.base import core
from paddle.base.framework import in_dygraph_mode

class TestDataParallelGroup(unittest.TestCase):

    def _create_var(self, dtype, shape):
        if False:
            return 10
        return paddle.rand(shape=shape, dtype=dtype)

    def assign_group_by_size(self, *args):
        if False:
            i = 10
            return i + 15
        if in_dygraph_mode():
            return core.eager_assign_group_by_size(*args)

    def test_construct_group0(self):
        if False:
            i = 10
            return i + 15
        var_list = []
        var_list.append(self._create_var('float32', [2, 50]))
        var_list.append(self._create_var('float32', [2, 100]))
        var_list.append(self._create_var('float32', [2, 50]))
        var_list.append(self._create_var('float32', [2, 25]))
        res = self.assign_group_by_size(var_list, [False, False, False, False], [400])
        self.assertEqual([[0], [1], [2], [3]], res)

    def test_construct_group1(self):
        if False:
            while True:
                i = 10
        var_list = []
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        res = self.assign_group_by_size(var_list, [False, False, False, False, False, False], [400])
        self.assertEqual([[0, 2], [1, 3], [4], [5]], res)

    def test_construct_group2(self):
        if False:
            print('Hello World!')
        var_list = []
        var_list.append(self._create_var('float32', [2, 50]))
        var_list.append(self._create_var('float32', [2, 50]))
        var_list.append(self._create_var('float32', [2, 50]))
        var_list.append(self._create_var('float32', [2, 50]))
        res = self.assign_group_by_size(var_list, [False, False, False, False], [400, 800])
        self.assertEqual([[0], [1, 2], [3]], res)

    def test_construct_group3(self):
        if False:
            i = 10
            return i + 15
        var_list = []
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        res = self.assign_group_by_size(var_list, [False, False, False, False, False, False], [200, 400])
        self.assertEqual([[0], [1], [2, 4], [3, 5]], res)

    def test_construct_group4(self):
        if False:
            print('Hello World!')
        var_list = []
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        res = self.assign_group_by_size(var_list, [False, False, False, False, False, False], [0])
        self.assertEqual([[0], [1], [2], [3], [4], [5]], res)

    def test_construct_group5(self):
        if False:
            print('Hello World!')
        var_list = []
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        res = self.assign_group_by_size(var_list, [False, False, False, False, False, False], [10000])
        self.assertEqual([[0, 2, 4], [1, 3, 5]], res)

    def test_construct_group6(self):
        if False:
            while True:
                i = 10
        var_list = []
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        res = self.assign_group_by_size(var_list, [True, False, False, False, False, True], [400])
        self.assertEqual([[0], [1, 3], [2, 4], [5]], res)

    def test_construct_group7(self):
        if False:
            return 10
        var_list = []
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        var_list.append(self._create_var('float32', [1, 50]))
        var_list.append(self._create_var('float64', [1, 25]))
        res = self.assign_group_by_size(var_list, [True, False, False, False, False, True], [200, 400])
        self.assertEqual([[0], [1], [2], [3], [4], [5]], res)

    def test_construct_group8(self):
        if False:
            return 10
        var_list = []
        var_list.append(self._create_var('float32', [2, 25]))
        var_list.append(self._create_var('float32', [2, 100]))
        var_list.append(self._create_var('float32', [2, 50]))
        var_list.append(self._create_var('float32', [2, 25]))
        res = self.assign_group_by_size(var_list, [False, False, False, False], [400], [3, 0, 1, 2])
        self.assertEqual([[3, 0], [1], [2]], res)

    def test_construct_group9(self):
        if False:
            while True:
                i = 10
        var_list = []
        var_list.append(self._create_var('float32', [2, 25]))
        var_list.append(self._create_var('float32', [2, 25]))
        var_list.append(self._create_var('float32', [2, 25]))
        var_list.append(self._create_var('float32', [2, 1000]))
        res = self.assign_group_by_size(var_list, [False, False, False, True], [300], [1, 0, 2, 3])
        self.assertEqual([[1, 0], [3], [2]], res)
if __name__ == '__main__':
    unittest.main()
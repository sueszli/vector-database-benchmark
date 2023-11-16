import unittest
from collections import OrderedDict
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core
from paddle.base.dygraph.base import to_variable

class MyLayer(paddle.nn.Layer):

    def __init__(self, name_scope):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name_scope)

    def forward(self, inputs):
        if False:
            print('Hello World!')
        x = F.relu(inputs)
        x = paddle.multiply(x, x)
        x = paddle.sum(x)
        return [x]

class TestImperativeParallelCoalesceSplit(unittest.TestCase):

    def test_coalesce_split(self):
        if False:
            for i in range(10):
                print('nop')
        from paddle.distributed.parallel import _coalesce_tensors, _split_tensors
        with base.dygraph.guard():
            test_layer = MyLayer('test_layer')
            strategy = core.ParallelStrategy()
            test_layer = paddle.DataParallel(test_layer, strategy)
            vars = []
            vars.append(to_variable(np.random.random([2, 3]).astype('float32')))
            vars.append(to_variable(np.random.random([4, 9]).astype('float32')))
            vars.append(to_variable(np.random.random([10, 1]).astype('float32')))
            var_groups = OrderedDict()
            var_groups.setdefault(0, vars)
            orig_var_shapes = []
            for var in vars:
                orig_var_shapes.append(var.shape)
            coalesced_vars = _coalesce_tensors(var_groups)
            _split_tensors(coalesced_vars)
            for (orig_var_shape, var) in zip(orig_var_shapes, vars):
                self.assertEqual(orig_var_shape, var.shape)

    def test_reshape_inplace(self):
        if False:
            i = 10
            return i + 15
        from paddle.distributed.parallel import _reshape_inplace
        with base.dygraph.guard():
            test_layer = MyLayer('test_layer')
            strategy = core.ParallelStrategy()
            test_layer = paddle.DataParallel(test_layer, strategy)
            ori_shape = [2, 25]
            new_shape = [5, 10]
            x_data = np.random.random(ori_shape).astype('float32')
            x = to_variable(x_data)
            _reshape_inplace(x, new_shape)
            self.assertEqual(x.shape, new_shape)
if __name__ == '__main__':
    unittest.main()
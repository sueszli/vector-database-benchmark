import unittest
import numpy as np
import paddle

class TestEagerTraceOp(unittest.TestCase):

    def test_branches(self):
        if False:
            i = 10
            return i + 15
        data = np.random.random([1, 1]).astype(np.float32)
        x = paddle.to_tensor(data)
        paddle.base.framework._dygraph_tracer().trace_op('broadcast_tensors', {'X': [x, x], 'Out': [x, x]}, {'Out': [x, x]}, {})
        paddle.base.framework._dygraph_tracer().trace_op('scale', {'X': x}, {'Out': x}, {'scale': 0.5})
        scale = paddle.to_tensor(np.random.random([1]).astype(np.float32))
        paddle.base.framework._dygraph_tracer().trace_op('instance_norm', {'Scale': [scale], 'X': [x]}, {'Y': [x]}, {})
if __name__ == '__main__':
    unittest.main()
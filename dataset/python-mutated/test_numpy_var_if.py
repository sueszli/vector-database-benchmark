import unittest
import numpy as np
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot.psdb import check_no_breakgraph, check_no_fallback
from paddle.jit.sot.utils import ENV_MIN_GRAPH_SIZE
ENV_MIN_GRAPH_SIZE.set(-1)

@check_no_breakgraph
@check_no_fallback
def forward(x, y):
    if False:
        i = 10
        return i + 15
    if x == 0:
        return y + 2
    else:
        return y * 2

@check_no_breakgraph
@check_no_fallback
def forward2(x, y):
    if False:
        for i in range(10):
            print('nop')
    if x == x:
        return y + 2
    else:
        return y * 2

class TestJumpWithNumpy(TestCaseBase):

    def test_jump(self):
        if False:
            while True:
                i = 10
        self.assert_results(forward, np.array([1]), paddle.to_tensor(2))
        self.assert_results(forward, np.array([0]), paddle.to_tensor(2))
        self.assert_results(forward2, np.array([0]), paddle.to_tensor(2))
if __name__ == '__main__':
    unittest.main()
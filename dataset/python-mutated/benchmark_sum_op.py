import unittest
import numpy as np
from benchmark import BenchmarkSuite

class TestSumOp(BenchmarkSuite):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'sum'
        self.customize_testcase()
        self.customize_fetch_list()

    def customize_fetch_list(self):
        if False:
            i = 10
            return i + 15
        '\n        customize fetch list, configure the wanted variables.\n        >>> self.fetch_list = ["Out"]\n        '
        self.fetch_list = ['Out']

    def customize_testcase(self):
        if False:
            return 10
        x0 = np.random.random((300, 400)).astype('float32')
        x1 = np.random.random((300, 400)).astype('float32')
        x2 = np.random.random((300, 400)).astype('float32')
        self.inputs = {'X': [('x0', x0), ('x1', x1), ('x2', x2)]}
        self.outputs = {'Out': x0 + x1 + x2}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        '\n        compare the output with customized output. In this case,\n        you should set the correct output by hands.\n        >>> self.outputs = {"Out": x0 + x1 + x2}\n        '
        self.check_output(atol=1e-08)

    def test_output_stability(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_stability()

    def test_timeit_output(self):
        if False:
            i = 10
            return i + 15
        '\n        perf the op, time cost will be averaged in iters.\n        output example\n        >>> One pass of (sum_op) at CPUPlace cost 0.000461330413818\n        >>> One pass of (sum_op) at CUDAPlace(0) cost 0.000556070804596\n        '
        self.timeit_output(iters=100)

    def test_timeit_grad(self):
        if False:
            i = 10
            return i + 15
        '\n        perf the op gradient, time cost will be averaged in iters.\n        output example\n        >>> One pass of (sum_grad_op) at CPUPlace cost 0.00279935121536\n        >>> One pass of (sum_grad_op) at CUDAPlace(0) cost 0.00500632047653\n        '
        self.timeit_grad(iters=100)
if __name__ == '__main__':
    unittest.main()
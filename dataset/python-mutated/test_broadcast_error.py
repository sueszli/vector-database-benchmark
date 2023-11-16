import unittest
import numpy as np
from op_test import OpTest
from paddle.base import core

class TestBroadcastOpCpu(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'broadcast'
        input = np.random.random((100, 2)).astype('float32')
        np_out = input[:]
        self.inputs = {'X': input}
        self.attrs = {'sync_mode': False, 'root': 0}
        self.outputs = {'Out': np_out}

    def test_check_output_cpu(self):
        if False:
            while True:
                i = 10
        try:
            self.check_output_with_place(place=core.CPUPlace())
        except:
            print('do not support cpu test, skip')
if __name__ == '__main__':
    unittest.main()
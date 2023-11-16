import unittest
import numpy as np
from pass_test import PassTest

class TestExpandZeroDimPass(PassTest):

    def init_input_data(self):
        if False:
            while True:
                i = 10
        self.feed_data = {'x': np.random.randint(-10, 10, []).astype('float32')}

    def build_program(self, builder, target):
        if False:
            while True:
                i = 10
        x = builder.create_input(self.nptype2cinntype(self.feed_data['x'].dtype), self.feed_data['x'].shape, 'x')
        out = builder.exp(x)
        return ([x], [out])

    def test_check_results(self):
        if False:
            i = 10
            return i + 15
        self.check_pass_outputs(pass_diff=0, test_passes=['ExpandZeroDim'], base_passes=[])
if __name__ == '__main__':
    unittest.main()
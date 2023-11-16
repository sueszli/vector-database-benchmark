import unittest
from pass_test import PassTest

class TestAutoCastPass(PassTest):

    def init_input_data(self):
        if False:
            return 10
        self.feed_data = {'x': self.random([4, 5, 6], 'float16')}

    def build_program(self, builder, target):
        if False:
            while True:
                i = 10
        x = builder.create_input(self.nptype2cinntype(self.feed_data['x'].dtype), self.feed_data['x'].shape, 'x')
        out = builder.exp(x)
        return ([x], [out])

    def test_check_results(self):
        if False:
            while True:
                i = 10
        self.check_pass_outputs(pass_diff=-2, test_passes=['AutoCast'], base_passes=['Decomposer'])
if __name__ == '__main__':
    unittest.main()
import unittest
from fusion_test import FusionTest

class TestGroup1(FusionTest):

    def init_input_data(self):
        if False:
            return 10
        self.feed_data = {}

    def build_program(self, builder, target):
        if False:
            for i in range(10):
                print('nop')
        x = builder.fill_constant(dtype='float32', shape=[4, 5, 20, 20], value=1.0)
        y = builder.cast(builder.reduce_sum(x, dim=[2], keep_dim=False), 'float16')
        feed_list = []
        fetch_list = [y]
        return (feed_list, fetch_list)

    def test_check_results(self):
        if False:
            while True:
                i = 10
        self.check_fusion_outputs(group_size=1)
if __name__ == '__main__':
    unittest.main()
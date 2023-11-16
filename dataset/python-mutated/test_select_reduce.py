import unittest
from fusion_test import FusionTest

class TestGroup1(FusionTest):

    def init_input_data(self):
        if False:
            i = 10
            return i + 15
        self.feed_data = {'cond': self.random([1, 1, 100, 100], 'bool'), 'true_value': self.random([1, 1, 100, 100], 'float64'), 'false_value': self.random([1, 1, 100, 100], 'float64')}

    def build_program(self, builder, target):
        if False:
            return 10
        cond = builder.create_input(self.nptype2cinntype(self.feed_data['cond'].dtype), self.feed_data['cond'].shape, 'cond')
        true_value = builder.create_input(self.nptype2cinntype(self.feed_data['true_value'].dtype), self.feed_data['true_value'].shape, 'true_value')
        false_value = builder.create_input(self.nptype2cinntype(self.feed_data['false_value'].dtype), self.feed_data['false_value'].shape, 'false_value')
        var_1 = builder.select(cond, true_value, false_value)
        var_2 = builder.reduce_sum(var_1, dim=[2], keep_dim=False)
        feed_list = [cond, true_value, false_value]
        fetch_list = [var_2]
        return (feed_list, fetch_list)

    def test_check_results(self):
        if False:
            return 10
        self.check_fusion_outputs(group_size=1)
if __name__ == '__main__':
    unittest.main()
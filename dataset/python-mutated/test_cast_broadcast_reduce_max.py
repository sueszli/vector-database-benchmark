import unittest
from fusion_test import FusionTest

class TestGroup1(FusionTest):

    def init_input_data(self):
        if False:
            while True:
                i = 10
        self.feed_data = {'eager_in_tmp_8': self.random([32, 1, 1, 128], 'float32')}

    def build_program(self, builder, target):
        if False:
            while True:
                i = 10
        eager_in_tmp_8 = builder.create_input(self.nptype2cinntype(self.feed_data['eager_in_tmp_8'].dtype), self.feed_data['eager_in_tmp_8'].shape, 'eager_in_tmp_8')
        var_15 = builder.cast(eager_in_tmp_8, dtype='float16')
        var_73 = builder.broadcast_to(var_15, broadcast_axes=[0, 1, 2, 3], out_shape=[32, 12, 128, 128])
        var_55 = builder.cast(var_73, dtype='float32')
        var_76 = builder.reduce_max(var_55, dim=[3], keep_dim=False)
        return ([eager_in_tmp_8], [var_15, var_76])

    def test_check_results(self):
        if False:
            while True:
                i = 10
        self.check_fusion_outputs(group_size=2)

class TestGroup2(FusionTest):

    def init_input_data(self):
        if False:
            i = 10
            return i + 15
        self.feed_data = {'eager_in_tmp_8': self.random([32, 1, 1, 128], 'float32')}

    def build_program(self, builder, target):
        if False:
            return 10
        eager_in_tmp_8 = builder.create_input(self.nptype2cinntype(self.feed_data['eager_in_tmp_8'].dtype), self.feed_data['eager_in_tmp_8'].shape, 'eager_in_tmp_8')
        var_15 = builder.cast(eager_in_tmp_8, dtype='float16')
        var_73 = builder.broadcast_to(var_15, broadcast_axes=[0, 1, 2, 3], out_shape=[32, 12, 128, 128])
        var_55 = builder.cast(var_73, dtype='float32')
        var_76 = builder.reduce_max(var_55, dim=[3], keep_dim=False)
        return ([eager_in_tmp_8], [var_76])

    def test_check_results(self):
        if False:
            while True:
                i = 10
        self.check_fusion_outputs(group_size=1)
if __name__ == '__main__':
    unittest.main()
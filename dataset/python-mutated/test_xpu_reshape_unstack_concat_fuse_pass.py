import unittest
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestReshapeUnstackConcatFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['reshape2', 'slice', 'reshape2', 'unstack', 'concat', 'reshape2', 'transpose2', 'split'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            while True:
                i = 10
        reshape_x_shape = [4, 48, 2, 16, 4096]
        reshape_op = OpConfig('reshape2', inputs={'X': ['reshape_x']}, outputs={'Out': ['reshape_out'], 'XShape': ['reshape_xshape']}, shape=[4, -1, 48, 2, 16, 4096])
        unstack_op = OpConfig('unstack', inputs={'X': ['reshape_out']}, outputs={'Y': ['unstakc_out0', 'unstakc_out1', 'unstakc_out2', 'unstakc_out3']}, axis=0, num=4)
        concat_op = OpConfig('concat', inputs={'X': ['unstakc_out0', 'unstakc_out1', 'unstakc_out2', 'unstakc_out3']}, outputs={'Out': ['concat_out']}, axis=-2)
        slice_0s = []
        reshape_0s = []
        slice_1s = []
        reshape_1s = []
        transposes = []
        out_names = []
        for i in range(48):
            slice_0_op = OpConfig('slice', inputs={'Input': ['concat_out']}, outputs={'Out': ['slice_0_' + str(i) + '_out']}, starts=[i], ends=[i + 1], axes=[1], decrease_axis=[])
            slice_0s.append(slice_0_op)
            reshape_0_op = OpConfig('reshape2', inputs={'X': ['slice_0_' + str(i) + '_out']}, outputs={'Out': ['reshape_0_' + str(i) + '_out'], 'XShape': ['reshape_0_' + str(i) + '_xshape']}, shape=[-1, 2, 64, 4, 1024])
            reshape_0s.append(reshape_0_op)
            slice_1_op = OpConfig('slice', inputs={'Input': ['reshape_0_' + str(i) + '_out']}, outputs={'Out': ['slice_1_' + str(i) + '_out']}, starts=[1], ends=[2], axes=[3], decrease_axis=[3])
            slice_1s.append(slice_1_op)
            reshape_1_op = OpConfig('reshape2', inputs={'X': ['slice_1_' + str(i) + '_out']}, outputs={'Out': ['reshape_1_' + str(i) + '_out'], 'XShape': ['reshape_1_' + str(i) + '_xshape']}, shape=[-1, 2, 64, 16, 64])
            reshape_1s.append(reshape_1_op)
            transpose_op = OpConfig('transpose2', inputs={'X': ['reshape_1_' + str(i) + '_out']}, outputs={'Out': ['transpose_' + str(i) + '_out'], 'XShape': ['transpose_' + str(i) + '_xshape']}, axis=[1, 0, 3, 2, 4])
            transposes.append(transpose_op)
            out_names.append('transpose_' + str(i) + '_out')
        ops = [reshape_op, unstack_op, concat_op]
        ops.extend(slice_0s)
        ops.extend(reshape_0s)
        ops.extend(slice_1s)
        ops.extend(reshape_1s)
        ops.extend(transposes)
        program_config = ProgramConfig(ops=ops, weights={}, inputs={'reshape_x': TensorConfig(shape=reshape_x_shape)}, outputs=out_names)
        return program_config

    def test(self):
        if False:
            print('Hello World!')
        self.run_and_statis(quant=False, max_examples=1, min_success_num=1, passes=['reshape_unstack_concat_fuse_pass'])
if __name__ == '__main__':
    unittest.main()
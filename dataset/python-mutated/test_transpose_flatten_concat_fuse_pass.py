import unittest
import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestTransposeFlattenConcatFusePass(PassAutoScanTest):
    """
        x_1_var              x_2_var
          |                     |
      transpose2            transpose2
          |                     |
       flatten2              flatten2
          \\                     /
    flatten2_out_var    flatten2_out_var
              \\              /
                   concat
    """

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config(use_gpu=True)
        yield (config, ['fusion_transpose_flatten_concat'], (1e-05, 1e-05))

    def is_program_valid(self, prog_config):
        if False:
            return 10
        concat_axis = prog_config.ops[-1].attrs['axis']
        ops_num = len(prog_config.ops) - 1
        if ops_num % 2 != 0:
            return False
        input_num = ops_num // 2
        flatten_shape = 0
        x_trans_axis = prog_config.ops[0].attrs['axis']
        x_flatten_axis = prog_config.ops[1].attrs['axis']
        for i in range(input_num):
            input_name = 'transpose2_x' + str(i)
            input_shape = prog_config.inputs[input_name].shape
            trans_axis = prog_config.ops[i * 2].attrs['axis']
            if x_trans_axis != trans_axis:
                return False
            input_shape = [input_shape[j] for j in trans_axis]
            flatten_axis = prog_config.ops[i * 2 + 1].attrs['axis']
            if x_flatten_axis != flatten_axis:
                return False
            flatten_shape1 = flatten_shape2 = 1
            for j in range(len(input_shape)):
                if j < flatten_axis:
                    flatten_shape1 *= input_shape[j]
                else:
                    flatten_shape2 *= input_shape[j]
            if concat_axis == 0:
                if i == 0:
                    flatten_shape = flatten_shape2
                elif flatten_shape != flatten_shape2:
                    return False
            elif i == 0:
                flatten_shape = flatten_shape1
            elif flatten_shape != flatten_shape1:
                return False
        return True

    def sample_program_config(self, draw):
        if False:
            while True:
                i = 10
        times = draw(st.integers(min_value=1, max_value=6))
        concat_axis = draw(st.integers(min_value=0, max_value=1))
        ops = []
        concat_input = []
        inputs = {}
        x_shape_rank = draw(st.integers(min_value=2, max_value=5))
        trans_axis = list(range(x_shape_rank))
        for j in range(x_shape_rank - 1):
            if draw(st.booleans()):
                (trans_axis[j], trans_axis[-1]) = (trans_axis[-1], trans_axis[j])
        flatten_axis = draw(st.integers(min_value=0, max_value=x_shape_rank - 1))
        for i in range(times):
            x_shape = draw(st.lists(st.integers(min_value=1, max_value=10), min_size=x_shape_rank, max_size=x_shape_rank))
            str_i = str(i)
            transpose_op = OpConfig('transpose2', inputs={'X': ['transpose2_x' + str_i]}, axis=trans_axis, outputs={'Out': ['trans_out' + str_i], 'XShape': ['trans_shape' + str_i]})
            ops.append(transpose_op)
            flatten_op = OpConfig('flatten2', inputs={'X': ['trans_out' + str_i]}, axis=flatten_axis, outputs={'Out': ['flatten2_out' + str_i], 'XShape': ['xshape' + str_i]})
            concat_input.append('flatten2_out' + str_i)
            ops.append(flatten_op)
            inputs['transpose2_x' + str_i] = TensorConfig(shape=x_shape)
        concat_op = OpConfig('concat', inputs={'X': concat_input, 'AxisTensor': []}, outputs={'Out': ['concat_out']}, axis=concat_axis)
        ops.append(concat_op)
        program_config = ProgramConfig(ops=ops, weights={}, inputs=inputs, outputs=ops[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            return 10
        self.run_and_statis(quant=False, max_examples=300, passes=['transpose_flatten_concat_fuse_pass'])
if __name__ == '__main__':
    unittest.main()
import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestEmbeddingWithEltwiseAddXPUFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['embedding_with_eltwise_add_xpu'], (0.001, 0.001))

    def sample_program_config(self, draw):
        if False:
            print('Hello World!')
        lookup_table_num = draw(st.sampled_from([2, 3, 4]))
        print('lookup_table_num: ', lookup_table_num)
        ids_shape = draw(st.sampled_from([[1, 32]]))
        w_shape = draw(st.sampled_from([[1000, 32]]))
        padding_idx = draw(st.sampled_from([-1]))
        axis = draw(st.sampled_from([-1]))

        def gen_lookup_table_ops():
            if False:
                print('Hello World!')
            lookup_table_op_config_list = []
            lookup_table_op_0 = OpConfig('lookup_table_v2', inputs={'Ids': ['lookup_table_ids_0'], 'W': ['lookup_table_w_0']}, outputs={'Out': ['lookup_table_out_0']}, padding_idx=padding_idx)
            lookup_table_op_1 = OpConfig('lookup_table_v2', inputs={'Ids': ['lookup_table_ids_1'], 'W': ['lookup_table_w_1']}, outputs={'Out': ['lookup_table_out_1']}, padding_idx=padding_idx)
            lookup_table_ops_list = [lookup_table_op_0, lookup_table_op_1]
            if lookup_table_num >= 3:
                lookup_table_op_2 = OpConfig('lookup_table_v2', inputs={'Ids': ['lookup_table_ids_2'], 'W': ['lookup_table_w_2']}, outputs={'Out': ['lookup_table_out_2']}, padding_idx=padding_idx)
                lookup_table_ops_list.append(lookup_table_op_2)
            if lookup_table_num >= 4:
                lookup_table_op_3 = OpConfig('lookup_table_v2', inputs={'Ids': ['lookup_table_ids_3'], 'W': ['lookup_table_w_3']}, outputs={'Out': ['lookup_table_out_3']}, padding_idx=padding_idx)
                lookup_table_ops_list.append(lookup_table_op_3)
            return lookup_table_ops_list
        add_op_num = lookup_table_num - 1

        def gen_eltwise_add_ops():
            if False:
                return 10
            add_op_0 = OpConfig('elementwise_add', inputs={'X': ['lookup_table_out_0'], 'Y': ['lookup_table_out_1']}, outputs={'Out': ['add_op_0_out']}, axis=axis)
            add_op_list = [add_op_0]
            if add_op_num >= 2:
                add_op_1 = OpConfig('elementwise_add', inputs={'X': ['add_op_0_out'], 'Y': ['lookup_table_out_2']}, outputs={'Out': ['add_op_1_out']}, axis=axis)
                add_op_list.append(add_op_1)
            if add_op_num >= 3:
                add_op_2 = OpConfig('elementwise_add', inputs={'X': ['add_op_1_out'], 'Y': ['lookup_table_out_3']}, outputs={'Out': ['add_op_2_out']}, axis=axis)
                add_op_list.append(add_op_2)
            return add_op_list
        lookup_table_op_list = gen_lookup_table_ops()
        add_op_list = gen_eltwise_add_ops()
        ops = []
        ops.extend(lookup_table_op_list)
        ops.extend(add_op_list)

        def generate_input(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return np.random.randint(0, w_shape[0], ids_shape).astype(np.int64)

        def gen_lookup_table_inputs_data(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            inputs = {}
            for i in range(lookup_table_num):
                input_name = f'lookup_table_ids_{i}'
                inputs[input_name] = TensorConfig(data_gen=partial(generate_input))
            return inputs
        inputs = gen_lookup_table_inputs_data()

        def gen_lookup_table_weights_data():
            if False:
                i = 10
                return i + 15
            weights = {}
            for i in range(lookup_table_num):
                w_name = f'lookup_table_w_{i}'
                weights[w_name] = TensorConfig(shape=w_shape)
            return weights
        weights = gen_lookup_table_weights_data()
        program_config = ProgramConfig(ops=ops, weights=weights, inputs=inputs, outputs=add_op_list[-1].outputs['Out'])
        return program_config

    def test(self):
        if False:
            i = 10
            return i + 15
        self.run_and_statis(quant=False, max_examples=3, min_success_num=3, passes=['embedding_with_eltwise_add_xpu_fuse_pass'])
if __name__ == '__main__':
    unittest.main()
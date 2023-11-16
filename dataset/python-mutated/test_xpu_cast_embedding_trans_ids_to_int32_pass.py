import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestXpuCastEmbeddingTransIdsToInt32Pass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        if False:
            print('Hello World!')
        config = self.create_inference_config(use_xpu=True)
        yield (config, ['cast', 'lookup_table_v2'], (1e-05, 1e-05))

    def sample_program_config(self, draw):
        if False:
            return 10
        ids_shape = draw(st.integers(min_value=1, max_value=128))
        w_shape = draw(st.sampled_from([[20, 64], [32, 32], [23, 15], [24, 33]]))
        padding_idx = draw(st.sampled_from([-1]))
        cast_op = OpConfig('cast', inputs={'X': ['cast_input']}, outputs={'Out': ['cast_out']}, in_dtype=5, out_dtype=3)
        lookup_table_op = OpConfig('lookup_table_v2', inputs={'Ids': ['cast_out'], 'W': ['lookup_table_w']}, outputs={'Out': ['lookup_table_out']}, padding_idx=padding_idx)

        def gen_lookup_table_weights_data():
            if False:
                return 10
            weights = {}
            w_name = 'lookup_table_w'
            weights[w_name] = TensorConfig(shape=w_shape)
            return weights

        def generate_cast_input(*args, **kwargs):
            if False:
                return 10
            return np.random.randint(0, w_shape[0], ids_shape).astype(np.float32)

        def gen_input_data(*args, **kwargs):
            if False:
                print('Hello World!')
            inputs = {}
            input_name = 'cast_input'
            inputs[input_name] = TensorConfig(data_gen=partial(generate_cast_input))
            return inputs
        inputs = gen_input_data()
        weights = gen_lookup_table_weights_data()
        program_config = ProgramConfig(ops=[cast_op, lookup_table_op], weights=weights, inputs=inputs, outputs=['lookup_table_out'])
        return program_config

    def test(self):
        if False:
            return 10
        self.run_and_statis(quant=False, max_examples=25, passes=['cast_embedding_trans_ids_to_int32_pass'])
if __name__ == '__main__':
    unittest.main()
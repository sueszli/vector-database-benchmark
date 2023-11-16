import unittest
from functools import partial
import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

class TestSeqpoolCvmConcatFusePass(PassAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def sample_program_config(self, draw):
        if False:
            while True:
                i = 10
        is_test = True
        pooltype = 'SUM'
        pad_value1 = draw(st.floats())
        pad_value2 = draw(st.floats())
        pad_value3 = draw(st.floats())
        use_cvm = True
        axis = draw(st.sampled_from([1]))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input1():
            if False:
                return 10
            shape = [batch_size, 128, 6, 120]
            return np.random.random(shape).astype(np.float32)

        def generate_input2():
            if False:
                i = 10
                return i + 15
            shape = [batch_size, 2]
            return np.random.random(shape).astype(np.float32)

        def generate_input3():
            if False:
                print('Hello World!')
            return np.random.random([1, 768]).astype(np.float32)
        im2sequence_op = OpConfig(type='im2sequence', inputs={'X': ['input_data1']}, outputs={'Out': ['seq_out']}, attrs={'kernels': [6, 1], 'out_stride': [1, 1], 'paddings': [0, 0, 0, 0], 'strides': [1, 1]})
        sequence_pool_op1 = OpConfig(type='sequence_pool', inputs={'X': ['seq_out']}, outputs={'Out': ['seq_pool1_out'], 'MaxIndex': ['index1_out']}, attrs={'is_test': is_test, 'pooltype': pooltype, 'pad_value': pad_value1})
        sequence_pool_op2 = OpConfig(type='sequence_pool', inputs={'X': ['seq_out']}, outputs={'Out': ['seq_pool2_out'], 'MaxIndex': ['index2_out']}, attrs={'is_test': is_test, 'pooltype': pooltype, 'pad_value': pad_value2})
        sequence_pool_op3 = OpConfig(type='sequence_pool', inputs={'X': ['seq_out']}, outputs={'Out': ['seq_pool3_out'], 'MaxIndex': ['index3_out']}, attrs={'is_test': is_test, 'pooltype': pooltype, 'pad_value': pad_value3})
        cvm_op1 = OpConfig(type='cvm', inputs={'X': ['seq_pool1_out'], 'CVM': ['input_data2']}, outputs={'Y': ['cvm1_out']}, attrs={'use_cvm': use_cvm})
        cvm_op2 = OpConfig(type='cvm', inputs={'X': ['seq_pool2_out'], 'CVM': ['input_data2']}, outputs={'Y': ['cvm2_out']}, attrs={'use_cvm': use_cvm})
        cvm_op3 = OpConfig(type='cvm', inputs={'X': ['seq_pool3_out'], 'CVM': ['input_data2']}, outputs={'Y': ['cvm3_out']}, attrs={'use_cvm': use_cvm})
        concat_op = OpConfig(type='concat', inputs={'X': ['cvm1_out', 'cvm2_out', 'cvm3_out']}, outputs={'Out': ['concat_output']}, attrs={'axis': axis})
        model_net = [im2sequence_op, sequence_pool_op1, sequence_pool_op2, sequence_pool_op3, cvm_op1, cvm_op2, cvm_op3, concat_op]
        program_config = ProgramConfig(ops=model_net, weights={}, inputs={'input_data1': TensorConfig(data_gen=partial(generate_input1)), 'input_data2': TensorConfig(data_gen=partial(generate_input2)), 'input_data3': TensorConfig(data_gen=partial(generate_input3))}, outputs=['concat_output'])
        return program_config

    def sample_predictor_configs(self, program_config):
        if False:
            for i in range(10):
                print('nop')
        config = self.create_inference_config()
        yield (config, ['im2sequence', 'fusion_seqpool_cvm_concat'], (1e-05, 1e-05))

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_statis(quant=False, passes=['seqpool_cvm_concat_fuse_pass'])
if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
from op_test import convert_uint16_to_float
from test_lookup_table_bf16_op import TestLookupTableBF16Op, TestLookupTableBF16OpIds4D, TestLookupTableBF16OpWIsSelectedRows, TestLookupTableBF16OpWIsSelectedRows4DIds, _lookup
import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

class TestLookupTableV2BF16Op(TestLookupTableBF16Op):

    def init_test(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'lookup_table_v2'
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = 4
        self.mkldnn_data_type = 'bfloat16'

class TestLookupTableV2BF16OpIds4D(TestLookupTableBF16OpIds4D):

    def init_test(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'lookup_table_v2'
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = (2, 4, 5)
        self.mkldnn_data_type = 'bfloat16'

class TestLookupTableV2BF16OpWIsSelectedRows(TestLookupTableBF16OpWIsSelectedRows):

    def init_test(self):
        if False:
            print('Hello World!')
        self.op_type = 'lookup_table_v2'
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = 10

class TestLookupTableV2BF16OpWIsSelectedRows4DIds(TestLookupTableBF16OpWIsSelectedRows4DIds):

    def init_test(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'lookup_table_v2'
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = (3, 4, 5)

class TestLookupTableBF16OpWithPadding(TestLookupTableV2BF16Op):

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output_with_place(core.CPUPlace())

class TestLookupTableBF16OpIds4DPadding(TestLookupTableV2BF16OpIds4D):

    def test_check_output(self):
        if False:
            print('Hello World!')
        ids = self.inputs['Ids']
        flatten_idx = ids.flatten()
        padding_idx = np.random.choice(flatten_idx, 1)[0]
        self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output_with_place(core.CPUPlace())

class TestEmbeddingLayerBF16ConstantInitializer(unittest.TestCase):
    """
    Test embedding layer from input api and results for bfloat16
    """

    def set_initializer(self):
        if False:
            i = 10
            return i + 15
        self.initializer = paddle.nn.initializer.Constant(value=self.value)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'lookup_table_v2'
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = [4]
        self.w_shape = [10, 64]
        self.ids = np.random.randint(low=0, high=9, size=self.ids_shape).astype('int64')
        self.flat_ids = self.ids.flatten()
        self.value = 3.0
        self.w_fp32 = np.full(self.w_shape, self.value)
        self.place = base.CPUPlace()
        self.prog = base.Program()
        self.startup_prog = base.Program()
        self.set_initializer()
        with base.program_guard(self.prog, self.startup_prog):
            x = paddle.static.data(name='x', shape=[-1] + self.ids_shape, dtype='int64')
            self.emb = paddle.static.nn.embedding(input=x, size=self.w_shape, param_attr=base.ParamAttr(name='emb_weight', initializer=self.initializer), is_sparse=False, dtype='uint16')
        exe = base.Executor(self.place)
        exe.run(self.startup_prog)
        self.result = exe.run(self.prog, feed={'x': self.ids}, fetch_list=['emb_weight', self.emb])

    @test_with_pir_api
    def test_embedding_weights(self):
        if False:
            print('Hello World!')
        result = convert_uint16_to_float(self.result[0])
        np.testing.assert_array_equal(self.w_fp32, result)

    @test_with_pir_api
    def test_lookup_results(self):
        if False:
            return 10
        lookup_result = convert_uint16_to_float(self.result[1])
        lookup_ref = _lookup(self.w_fp32, self.ids, self.flat_ids, self.op_type)
        np.testing.assert_array_equal(lookup_result, lookup_ref)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
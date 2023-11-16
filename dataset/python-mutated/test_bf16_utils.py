import copy
import unittest
import paddle
from paddle import base
from paddle.base import core
from paddle.static import amp
paddle.enable_static()

class AMPTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.bf16_list = copy.copy(amp.bf16.amp_lists.bf16_list)
        self.fp32_list = copy.copy(amp.bf16.amp_lists.fp32_list)
        self.gray_list = copy.copy(amp.bf16.amp_lists.gray_list)
        self.amp_lists_ = None

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.amp_lists_.bf16_list, self.bf16_list)
        self.assertEqual(self.amp_lists_.fp32_list, self.fp32_list)
        self.assertEqual(self.amp_lists_.gray_list, self.gray_list)

    def test_amp_lists(self):
        if False:
            while True:
                i = 10
        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16()

    def test_amp_lists_1(self):
        if False:
            print('Hello World!')
        self.bf16_list.add('exp')
        self.fp32_list.remove('exp')
        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16({'exp'})

    def test_amp_lists_2(self):
        if False:
            for i in range(10):
                print('nop')
        self.fp32_list.remove('tan')
        self.bf16_list.add('tan')
        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16({'tan'})

    def test_amp_lists_3(self):
        if False:
            return 10
        self.bf16_list.add('lstm')
        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16({'lstm'})

    def test_amp_lists_4(self):
        if False:
            return 10
        self.bf16_list.remove('matmul_v2')
        self.fp32_list.add('matmul_v2')
        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(custom_fp32_list={'matmul_v2'})

    def test_amp_lists_5(self):
        if False:
            print('Hello World!')
        self.fp32_list.add('matmul_v2')
        self.bf16_list.remove('matmul_v2')
        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(custom_fp32_list={'matmul_v2'})

    def test_amp_lists_6(self):
        if False:
            return 10
        self.fp32_list.add('lstm')
        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(custom_fp32_list={'lstm'})

    def test_amp_lists_7(self):
        if False:
            for i in range(10):
                print('nop')
        self.fp32_list.add('reshape2')
        self.gray_list.remove('reshape2')
        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(custom_fp32_list={'reshape2'})

    def test_amp_list_8(self):
        if False:
            return 10
        self.bf16_list.add('reshape2')
        self.gray_list.remove('reshape2')
        self.amp_lists_ = amp.bf16.AutoMixedPrecisionListsBF16(custom_bf16_list={'reshape2'})

class AMPTest2(unittest.TestCase):

    def test_amp_lists_(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, amp.bf16.AutoMixedPrecisionListsBF16, {'lstm'}, {'lstm'})

    def test_find_op_index(self):
        if False:
            print('Hello World!')
        block = base.default_main_program().global_block()
        op_desc = core.OpDesc()
        idx = amp.fp16_utils.find_op_index(block.desc, op_desc)
        assert idx == -1

    def test_is_in_fp32_varnames(self):
        if False:
            print('Hello World!')
        block = base.default_main_program().global_block()
        var1 = block.create_var(name='X', shape=[3], dtype='float32')
        var2 = block.create_var(name='Y', shape=[3], dtype='float32')
        var3 = block.create_var(name='Z', shape=[3], dtype='float32')
        op1 = block.append_op(type='abs', inputs={'X': [var1]}, outputs={'Out': [var2]})
        op2 = block.append_op(type='abs', inputs={'X': [var2]}, outputs={'Out': [var3]})
        amp_lists_1 = amp.bf16.AutoMixedPrecisionListsBF16(custom_fp32_varnames={'X'})
        assert amp.bf16.amp_utils._is_in_fp32_varnames(op1, amp_lists_1)
        amp_lists_2 = amp.bf16.AutoMixedPrecisionListsBF16(custom_fp32_varnames={'Y'})
        assert amp.bf16.amp_utils._is_in_fp32_varnames(op2, amp_lists_2)
        assert amp.bf16.amp_utils._is_in_fp32_varnames(op1, amp_lists_2)

    def test_find_true_post_op(self):
        if False:
            return 10
        block = base.default_main_program().global_block()
        var1 = block.create_var(name='X', shape=[3], dtype='float32')
        var2 = block.create_var(name='Y', shape=[3], dtype='float32')
        var3 = block.create_var(name='Z', shape=[3], dtype='float32')
        op1 = block.append_op(type='abs', inputs={'X': [var1]}, outputs={'Out': [var2]})
        op2 = block.append_op(type='abs', inputs={'X': [var2]}, outputs={'Out': [var3]})
        res = amp.bf16.amp_utils.find_true_post_op(block.ops, op1, 'Y')
        assert res == [op2]

    def test_find_true_post_op_with_search_all(self):
        if False:
            return 10
        program = base.Program()
        block = program.current_block()
        startup_block = base.default_startup_program().global_block()
        var1 = block.create_var(name='X', shape=[3], dtype='float32')
        var2 = block.create_var(name='Y', shape=[3], dtype='float32')
        inititializer_op = startup_block._prepend_op(type='fill_constant', outputs={'Out': var1}, attrs={'shape': var1.shape, 'dtype': var1.dtype, 'value': 1.0})
        op1 = block.append_op(type='abs', inputs={'X': [var1]}, outputs={'Out': [var2]})
        result = amp.bf16.amp_utils.find_true_post_op(block.ops, inititializer_op, 'X', search_all=False)
        assert len(result) == 0
        result = amp.bf16.amp_utils.find_true_post_op(block.ops, inititializer_op, 'X', search_all=True)
        assert result == [op1]
if __name__ == '__main__':
    unittest.main()
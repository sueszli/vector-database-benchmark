import unittest
from paddle.base import core
from paddle.base.framework import Program, default_startup_program
main_program = default_startup_program()

class TestOperator(unittest.TestCase):

    def test_error_type(self):
        if False:
            return 10
        block = main_program._create_block()
        try:
            block.append_op()
            self.assertFail()
        except ValueError as v_err:
            self.assertEqual(str(v_err), '`type` to initialized an Operator can not be None.')
        try:
            block.append_op(type='no_such_op')
            self.assertFail()
        except ValueError as a_err:
            self.assertEqual(str(a_err), 'Operator "no_such_op" has not been registered.')

    def test_op_desc_creation(self):
        if False:
            return 10
        program = Program()
        block = program.current_block()
        mul_x = block.create_var(dtype='float32', shape=[5, 10], lod_level=0, name='mul.x')
        mul_y = block.create_var(dtype='float32', shape=[10, 8], lod_level=0, name='mul.y')
        mul_out = block.create_var(dtype='float32', shape=[5, 8], lod_level=0, name='mul.out')
        mul_op = block.append_op(type='mul', inputs={'X': [mul_x], 'Y': mul_y}, outputs={'Out': [mul_out]}, attrs={'x_num_col_dims': 1})
        self.assertNotEqual(str(mul_op), '')
        self.assertEqual(mul_op.type, 'mul')
        self.assertEqual(mul_op.input_names, ['X', 'Y'])
        self.assertEqual(mul_op.input('X'), ['mul.x'])
        self.assertEqual(mul_op.input('Y'), ['mul.y'])
        self.assertEqual(mul_op.output_names, ['Out'])
        self.assertEqual(mul_op.output('Out'), ['mul.out'])
        self.assertEqual(set(mul_op.attr_names), {'x_num_col_dims', 'y_num_col_dims', 'op_role', 'op_role_var', 'op_namescope', 'op_callstack', 'op_device', 'with_quant_attr'})
        self.assertEqual(mul_op.has_attr('x_num_col_dims'), True)
        self.assertEqual(mul_op.attr_type('x_num_col_dims'), core.AttrType.INT)
        self.assertEqual(mul_op.attr('x_num_col_dims'), 1)
        self.assertEqual(mul_op.has_attr('y_num_col_dims'), True)
        self.assertEqual(mul_op.attr_type('y_num_col_dims'), core.AttrType.INT)
        self.assertEqual(mul_op.attr('y_num_col_dims'), 1)
        self.assertEqual(mul_op.idx, 0)
        self.assertEqual(mul_out.op, mul_op)
        mul_op.desc.remove_input('X')
        self.assertEqual(mul_op.input_names, ['Y'])

    def test_mult_input(self):
        if False:
            print('Hello World!')
        program = Program()
        block = program.current_block()
        sum_x1 = block.create_var(dtype='int', shape=[3, 4], lod_level=0, name='sum.x1')
        sum_x2 = block.create_var(dtype='int', shape=[3, 4], lod_level=0, name='sum.x2')
        sum_x3 = block.create_var(dtype='int', shape=[3, 4], lod_level=0, name='sum.x3')
        sum_out = block.create_var(dtype='int', shape=[3, 4], lod_level=0, name='sum.out')
        sum_op = block.append_op(type='sum', inputs={'X': [sum_x1, sum_x2, sum_x3]}, outputs={'Out': sum_out})
        self.assertEqual(sum_op.type, 'sum')
        self.assertEqual(sum_op.input_names, ['X'])
        self.assertEqual(sum_op.input('X'), ['sum.x1', 'sum.x2', 'sum.x3'])
        self.assertEqual(sum_op.output_names, ['Out'])
        self.assertEqual(sum_op.output('Out'), ['sum.out'])
        self.assertEqual(sum_op.idx, 0)
        self.assertEqual(sum_out.op, sum_op)
if __name__ == '__main__':
    unittest.main()
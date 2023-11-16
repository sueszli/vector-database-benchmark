import unittest
import paddle
from paddle import base
from paddle.base import core, framework

class TestInferNoNeedBufferSlots(unittest.TestCase):

    def net(self):
        if False:
            print('Hello World!')
        x1 = base.default_main_program().global_block().create_var(dtype='float32', shape=[1], lod_level=0, name='x1')
        x2 = base.default_main_program().global_block().create_var(dtype='float32', shape=[1], lod_level=0, name='x2')
        x = paddle.add(x1, x2)
        return x

    def test_infer_no_need_buffer_slots(self):
        if False:
            i = 10
            return i + 15
        program = framework.Program()
        startup_program = framework.Program()
        with base.program_guard(program, startup_program):
            loss = self.net()
            sgd = paddle.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)
        block = program.global_block()
        for (idx, op) in enumerate(block.ops):
            op_desc = op.desc
            inputs = {}
            for input_name in op_desc.input_names():
                inputs[input_name] = op_desc.input(input_name)
            outputs = {}
            for output_name in op_desc.output_names():
                outputs[output_name] = op_desc.output(output_name)
            attrs = {}
            for attr_name in op_desc.attr_names():
                attrs[attr_name] = op_desc.attr(attr_name)
            if idx == 0:
                self.assertEqual(core.infer_no_need_buffer_slots(op.type, inputs, outputs, attrs), set())
            elif idx == 1:
                self.assertEqual(core.infer_no_need_buffer_slots(op.type, inputs, outputs, attrs), set())
            else:
                self.assertEqual(core.infer_no_need_buffer_slots(op.type, inputs, outputs, attrs), {'Y', 'X'})
if __name__ == '__main__':
    unittest.main()
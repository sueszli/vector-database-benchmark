import os
import tempfile
import unittest
import paddle
import paddle.nn.functional as F

def getModelOp(model_path):
    if False:
        for i in range(10):
            print('nop')
    model_bytes = paddle.static.load_from_file(model_path)
    pg = paddle.static.deserialize_program(model_bytes)
    main_block = pg.desc.block(0)
    size = main_block.op_size()
    result = set()
    for i in range(0, size):
        result.add(main_block.op(i).type())
    return result

class WhileNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def forward(self, x):
        if False:
            while True:
                i = 10
        y = paddle.rand(shape=[1, 3, 4, 4])
        w1 = paddle.shape(y)[0]
        w2 = paddle.assign(paddle.shape(x)[0])
        while w2 != w1:
            x = F.avg_pool2d(x, kernel_size=3, padding=1, stride=2)
            w2 = paddle.shape(x)[0]
        return x + y

class ForNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        y = paddle.randint(low=0, high=5, shape=[1], dtype='int32')
        z = paddle.randint(low=0, high=5, shape=[1], dtype='int32')
        for i in range(0, z):
            x = x + i
        return x + y

class IfElseNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        y = paddle.to_tensor([5])
        if x > y:
            x = x + 1
        else:
            x = x - 1
        return x

class TestConditionalOp(unittest.TestCase):

    def test_while_op(self):
        if False:
            return 10
        paddle.disable_static()
        net = WhileNet()
        net = paddle.jit.to_static(net, input_spec=[paddle.static.InputSpec(shape=[1, 3, 8, 8], dtype='float32')])
        root_path = tempfile.TemporaryDirectory()
        model_file = os.path.join(root_path.name, 'while_net')
        paddle.jit.save(net, model_file)
        right_pdmodel = {'uniform_random', 'shape', 'slice', 'not_equal', 'while', 'elementwise_add'}
        paddle.enable_static()
        pdmodel = getModelOp(model_file + '.pdmodel')
        self.assertTrue(len(right_pdmodel.difference(pdmodel)) == 0, 'The while op is pruned by mistake.')
        root_path.cleanup()

    def test_for_op(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        net = ForNet()
        net = paddle.jit.to_static(net, input_spec=[paddle.static.InputSpec(shape=[1], dtype='int32')])
        root_path = tempfile.TemporaryDirectory()
        model_file = os.path.join(root_path.name, 'for_net')
        paddle.jit.save(net, model_file)
        right_pdmodel = {'randint', 'fill_constant', 'cast', 'less_than', 'while', 'elementwise_add'}
        paddle.enable_static()
        pdmodel = getModelOp(model_file + '.pdmodel')
        self.assertTrue(len(right_pdmodel.difference(pdmodel)) == 0, 'The for op is pruned by mistake.')
        root_path.cleanup()

    def test_if_op(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        net = IfElseNet()
        net = paddle.jit.to_static(net, input_spec=[paddle.static.InputSpec(shape=[1], dtype='int32')])
        root_path = tempfile.TemporaryDirectory()
        model_file = os.path.join(root_path.name, 'if_net')
        paddle.jit.save(net, model_file)
        right_pdmodel = {'assign_value', 'greater_than', 'cast', 'conditional_block', 'logical_not', 'select_input'}
        paddle.enable_static()
        pdmodel = getModelOp(model_file + '.pdmodel')
        self.assertTrue(len(right_pdmodel.difference(pdmodel)) == 0, 'The if op is pruned by mistake.')
        root_path.cleanup()
if __name__ == '__main__':
    unittest.main()
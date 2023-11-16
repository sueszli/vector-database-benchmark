import unittest
import nets
import paddle
from paddle import base
from paddle.base.framework import Program

def conv_block(input, num_filter, groups, dropouts):
    if False:
        i = 10
        return i + 15
    return nets.img_conv_group(input=input, pool_size=2, pool_stride=2, conv_num_filter=[num_filter] * groups, conv_filter_size=3, conv_act='relu', conv_with_batchnorm=True, conv_batchnorm_drop_rate=dropouts, pool_type='max')

class TestLayer(unittest.TestCase):

    def test_batch_norm_layer(self):
        if False:
            return 10
        main_program = Program()
        startup_program = Program()
        with base.program_guard(main_program, startup_program):
            images = paddle.static.data(name='pixel', shape=[-1, 3, 48, 48], dtype='float32')
            hidden1 = paddle.static.nn.batch_norm(input=images)
            hidden2 = paddle.static.nn.fc(x=hidden1, size=128, activation='relu')
            paddle.static.nn.batch_norm(input=hidden2)
        print(str(main_program))

    def test_dropout_layer(self):
        if False:
            i = 10
            return i + 15
        main_program = Program()
        startup_program = Program()
        with base.program_guard(main_program, startup_program):
            images = paddle.static.data(name='pixel', shape=[-1, 3, 48, 48], dtype='float32')
            paddle.nn.functional.dropout(x=images, p=0.5)
        print(str(main_program))

    def test_img_conv_group(self):
        if False:
            i = 10
            return i + 15
        main_program = Program()
        startup_program = Program()
        with base.program_guard(main_program, startup_program):
            images = paddle.static.data(name='pixel', shape=[-1, 3, 48, 48], dtype='float32')
            conv1 = conv_block(images, 64, 2, [0.3, 0])
            conv_block(conv1, 256, 3, [0.4, 0.4, 0])
        print(str(main_program))

    def test_elementwise_add_with_act(self):
        if False:
            i = 10
            return i + 15
        main_program = Program()
        startup_program = Program()
        with base.program_guard(main_program, startup_program):
            image1 = paddle.static.data(name='pixel1', shape=[-1, 3, 48, 48], dtype='float32')
            image2 = paddle.static.data(name='pixel2', shape=[-1, 3, 48, 48], dtype='float32')
            paddle.nn.functional.relu(paddle.add(x=image1, y=image2))
        print(main_program)
if __name__ == '__main__':
    unittest.main()
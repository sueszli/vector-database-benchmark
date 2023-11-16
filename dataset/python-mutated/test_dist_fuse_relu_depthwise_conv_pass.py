import os
import unittest
import numpy as np
from dist_pass_test_base import DistPassTestBase
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.passes import PassManager, new_pass
paddle.enable_static()

class ReluDepthwiseConvNet(nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.conv1 = nn.Conv2D(3, 9, (3, 3))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(9, 27, (3, 3), groups=9)

    def forward(self, x):
        if False:
            while True:
                i = 10
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = paddle.flatten(out, 1)
        return out

class TestFuseReluDepthwiseConvPass(DistPassTestBase):

    def init(self):
        if False:
            print('Hello World!')
        self.atol = 0.0001
        self.rtol = 0.0001

    def get_model(self, place, batch_size=32, image_shape=[3, 224, 224]):
        if False:
            print('Hello World!')
        image = paddle.static.data(shape=[batch_size] + image_shape, dtype='float32', name='image')
        model = ReluDepthwiseConvNet()
        pred_out = model(image)
        loss = paddle.mean(pred_out)
        optimizer = paddle.optimizer.Adam(learning_rate=0.001)
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.fuse_all_reduce_ops = False
        dist_strategy.without_graph_optimization = True
        fleet.init(is_collective=True, strategy=dist_strategy)
        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(loss)
        rank = paddle.distributed.get_rank()

        def reader():
            if False:
                print('Hello World!')
            seed = int(os.environ.get('SEED', 0))
            np.random.seed(seed + rank)
            for _ in range(10):
                image_np = np.random.random(size=image.shape).astype('float32')
                yield (image_np,)
        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        return (main_program, startup_program, [image], [loss], reader)

    def apply_passes(self, main_prog, startup_prog):
        if False:
            for i in range(10):
                print('nop')
        pass_manager = PassManager([new_pass('fuse_relu_depthwise_conv')])
        pass_manager.apply([main_prog], [startup_prog])
        print(pass_manager.names)
        op_type = []
        for op in main_prog.global_block().ops:
            if op.type == 'depthwise_conv2d':
                self.assertTrue(op.desc.attr('fuse_relu_before_depthwise_conv'))
            op_type.append(op.type)
        self.assertTrue('depthwise_conv2d' in op_type)

    def test_relu_depthwise_conv(self):
        if False:
            return 10
        self.check_main()
if __name__ == '__main__':
    unittest.main()
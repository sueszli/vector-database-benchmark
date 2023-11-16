import os
import unittest
import numpy as np
from dist_pass_test_base import DistPassTestBase
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.passes import PassManager, new_pass
paddle.enable_static()

class DemoNet(nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.conv1 = nn.Conv2D(3, 8, (3, 3), data_format='NHWC')
        self.bn1 = nn.BatchNorm2D(8, data_format='NHWC')
        self.relu = nn.ReLU()

    def forward(self, x):
        if False:
            return 10
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = paddle.flatten(out, 1)
        return out

class TestFuseAdamPass(DistPassTestBase):

    def init(self):
        if False:
            while True:
                i = 10
        self.atol = 0.0001
        self.rtol = 0.0001

    def get_model(self, place, batch_size=32, image_shape=[224, 224, 3]):
        if False:
            return 10
        image = paddle.static.data(shape=[batch_size] + image_shape, dtype='float32', name='image')
        model = DemoNet()
        pred_out = model(image)
        loss = paddle.mean(pred_out)
        optimizer = paddle.optimizer.Momentum(learning_rate=0.001)
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
            while True:
                i = 10
        pass_manager = PassManager([new_pass('fuse_optimizer')])
        pass_manager.apply([main_prog], [startup_prog])
        print(pass_manager.names)
        op_type = []
        for op in main_prog.global_block().ops:
            op_type.append(op.type)
            if op.type == 'momentum':
                self.assertTrue('@FUSEDVAR@_momentum_Param_batch_norm2d_0.b_0' in op.input('Param'))
                self.assertTrue('@FUSEDVAR@_momentum_Grad_batch_norm2d_0.b_0@GRAD' in op.input('Grad'))
        self.assertTrue('coalesce_tensor' in op_type)

    def test_fuse_adam(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_main()
if __name__ == '__main__':
    unittest.main()
import os
import unittest
import numpy as np
from dist_pass_test_base import DistPassTestBase
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.passes import PassManager, new_pass
paddle.enable_static()

class BatchNormAddActNet(nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.conv1 = nn.Conv2D(3, 8, (3, 3), data_format='NHWC')
        self.conv2 = nn.Conv2D(3, 8, (3, 3), data_format='NHWC')
        self.bn1 = nn.BatchNorm2D(8, data_format='NHWC')
        self.bn2 = nn.BatchNorm2D(8, data_format='NHWC')
        self.relu = nn.ReLU()

    def forward(self, x):
        if False:
            print('Hello World!')
        y = self.conv1(x)
        y = self.bn1(y)
        out = self.conv2(x)
        out = self.bn2(out) + y
        out = self.relu(out)
        out = paddle.flatten(out, 1)
        return out

class TestFuseBatchNormAddActPass(DistPassTestBase):

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        self.atol = 0.0001
        self.rtol = 0.0001

    def get_model(self, place, batch_size=32, image_shape=[224, 224, 3]):
        if False:
            i = 10
            return i + 15
        image = paddle.static.data(shape=[batch_size] + image_shape, dtype='float32', name='image')
        model = BatchNormAddActNet()
        pred_out = model(image)
        loss = paddle.mean(pred_out)
        optimizer = paddle.optimizer.Adam(learning_rate=0.001)
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.fuse_all_reduce_ops = False
        dist_strategy.without_graph_optimization = True
        dist_strategy.amp = True
        dist_strategy.amp_configs = {'init_loss_scaling': 32768, 'use_dynamic_loss_scaling': True}
        fleet.init(is_collective=True, strategy=dist_strategy)
        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(loss)
        rank = paddle.distributed.get_rank()

        def reader():
            if False:
                return 10
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
            return 10
        pass_manager = PassManager([new_pass('fuse_bn_add_act')])
        pass_manager.apply([main_prog], [startup_prog])
        print(pass_manager.names)
        op_type = []
        for op in main_prog.global_block().ops:
            op_type.append(op.type)
        self.assertTrue('fused_bn_add_activation' in op_type)
        self.assertTrue('fused_bn_add_activation_grad' in op_type)

    def test_fuse_bn_add_act(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_main()
if __name__ == '__main__':
    unittest.main()
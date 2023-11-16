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
            return 10
        super().__init__()
        self.conv1 = nn.Conv2D(3, 3, (3, 3), padding=1, data_format='NHWC')
        self.conv2 = nn.Conv2D(3, 3, (3, 3), padding=1, data_format='NHWC')
        self.conv3 = nn.Conv2D(3, 3, (3, 3), padding=1, data_format='NHWC')
        self.relu = nn.ReLU()

    def forward(self, x):
        if False:
            return 10
        out = self.conv1(x)
        out = self.conv2(out) + self.conv3(out)
        out = self.relu(out)
        out = paddle.flatten(out, 1)
        return out

class TestInplaceAddtoPass(DistPassTestBase):

    def init(self):
        if False:
            print('Hello World!')
        self.atol = 0.0
        self.rtol = 0.0
        paddle.base.set_flags({'FLAGS_max_inplace_grad_add': 8})

    def get_model(self, place, batch_size=32, image_shape=[224, 224, 3]):
        if False:
            while True:
                i = 10
        image = paddle.static.data(shape=[batch_size] + image_shape, dtype='float32', name='image')
        model = DemoNet()
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
                while True:
                    i = 10
            seed = int(os.environ.get('SEED', 0))
            np.random.seed(seed + rank)
            for _ in range(10):
                image_np = np.random.random(size=image.shape).astype('float32')
                yield (image_np,)
        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        print(main_program)
        return (main_program, startup_program, [image], [loss], reader)

    def apply_passes(self, main_prog, startup_prog):
        if False:
            return 10
        pass_manager = PassManager([new_pass('inplace_addto_op', {'use_cuda': True})])
        pass_manager.apply([main_prog], [startup_prog])
        print(pass_manager.names)
        conv2d_grad_attr = []
        for op in main_prog.global_block().ops:
            if op.type == 'conv2d_grad':
                conv2d_grad_attr.append(op.desc.attr('use_addto'))
        self.assertTrue(True in conv2d_grad_attr)

    def test_inplace_addto(self):
        if False:
            return 10
        self.check_main()
if __name__ == '__main__':
    unittest.main()
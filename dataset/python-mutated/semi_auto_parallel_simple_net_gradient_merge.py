import os
import numpy as np
from semi_auto_parallel_simple_net import DemoNet, TestSimpleNetForSemiAutoParallel
import paddle
import paddle.distributed as dist
from paddle import nn

class TestSimpleNetWithGradientMergeForSemiAutoParallel(TestSimpleNetForSemiAutoParallel):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        paddle.set_device(self._backend)
        self.init_single_card_net_result()

    def run_dynamic_gradient_merge(self, layer, shard_input=False):
        if False:
            while True:
                i = 10
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        loss_fn = nn.MSELoss()
        opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=layer.parameters())
        (image, label) = self.init_input_data()
        if shard_input:
            image = dist.shard_tensor(image, dist_attr=dist.DistAttr(mesh=self._mesh, sharding_specs=['x', None]))
        for i in range(2):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return (loss, layer.parameters())

    def init_single_card_net_result(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_random_seed(self._seed)
        (self.base_loss, self.base_parameters) = self.run_dynamic_gradient_merge(DemoNet('gradient_merge_demo'))

    def test_dp_demo_net(self):
        if False:
            print('Hello World!')
        self.set_random_seed(self._seed)
        (self.dp_loss, self.dp_parameters) = self.run_dynamic_gradient_merge(DemoNet('gradient_merge_dp_demo'), shard_input=True)
        self.check_tensor_eq(self.dp_loss, self.base_loss)
        self.check_tensor_eq(self.dp_loss, self.base_loss)
        for (param, param_base) in zip(self.dp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def test_mp_demo_net(self):
        if False:
            i = 10
            return i + 15
        self.set_random_seed(self._seed)
        mp_layer = dist.shard_layer(DemoNet('gradient_merge_mp_demo'), self._mesh, self.shard_fn)
        (self.mp_loss, self.mp_parameters) = self.run_dynamic_gradient_merge(mp_layer)
        self.check_tensor_eq(self.mp_loss, self.base_loss)
        for (param, param_base) in zip(self.mp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def run_test_case(self):
        if False:
            while True:
                i = 10
        self.test_dp_demo_net()
        self.test_mp_demo_net()
if __name__ == '__main__':
    TestSimpleNetWithGradientMergeForSemiAutoParallel().run_test_case()
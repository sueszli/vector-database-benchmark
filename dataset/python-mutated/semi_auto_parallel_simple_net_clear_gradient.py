import os
from semi_auto_parallel_simple_net import DemoNet, TestSimpleNetForSemiAutoParallel
import paddle
import paddle.distributed as dist
from paddle import nn

class TestSimpleNetWithClearGradientForSemiAutoParallel(TestSimpleNetForSemiAutoParallel):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        paddle.set_device(self._backend)

    def run_dynamic_clear_gradient(self, layer, shard_input=False):
        if False:
            print('Hello World!')
        loss_fn = nn.MSELoss()
        opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=layer.parameters())
        for _ in range(5):
            (image, label) = self.init_input_data()
            if shard_input:
                image = dist.shard_tensor(image, dist_attr=dist.DistAttr(mesh=self._mesh, sharding_specs=['x', None]))
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
            for param in layer.parameters():
                param.clear_gradient()
                param.clear_gradient(False)

    def test_demo_net(self):
        if False:
            i = 10
            return i + 15
        mp_layer = dist.shard_layer(DemoNet('clear_gradient_demo'), self._mesh, self.shard_fn)
        self.run_dynamic_clear_gradient(mp_layer)

    def run_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_demo_net()
if __name__ == '__main__':
    TestSimpleNetWithClearGradientForSemiAutoParallel().run_test_case()
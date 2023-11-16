import os
from semi_auto_parallel_simple_net import DemoNet, TestSimpleNetForSemiAutoParallel
import paddle
import paddle.distributed as dist
from paddle import nn

class TestSimpleNetWithZeroGradsForSemiAutoParallel(TestSimpleNetForSemiAutoParallel):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        paddle.set_device(self._backend)
        (self.image, self.label) = self.init_input_data()

    def run_dynamic_zero_grads(self, layer, shard_input=False):
        if False:
            i = 10
            return i + 15
        loss_fn = nn.MSELoss()
        (image, label) = self.init_input_data()
        if shard_input:
            image = dist.shard_tensor(image, dist_attr=dist.DistAttr(mesh=self._mesh, sharding_specs=['x', None]))
        out = layer(image)
        loss = loss_fn(out, label)
        loss.backward()
        for param in layer.parameters():
            param._zero_grads()

    def test_demo_net(self):
        if False:
            print('Hello World!')
        mp_layer = dist.shard_layer(DemoNet('zero_grads_demo'), self._mesh, self.shard_fn)
        self.run_dynamic_zero_grads(mp_layer)

    def run_test_case(self):
        if False:
            print('Hello World!')
        self.test_demo_net()
if __name__ == '__main__':
    TestSimpleNetWithZeroGradsForSemiAutoParallel().run_test_case()
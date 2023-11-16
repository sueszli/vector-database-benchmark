import os
from semi_auto_parallel_simple_net import DemoNet, TestSimpleNetForSemiAutoParallel
import paddle
import paddle.distributed as dist
from paddle import nn

class TestSimpleNetWithGradApiForSemiAutoParallel(TestSimpleNetForSemiAutoParallel):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        paddle.set_device(self._backend)

    def run_dynamic_grad_api(self, layer, shard_input=False):
        if False:
            return 10
        loss_fn = nn.MSELoss()
        (image, label) = self.init_input_data()
        if shard_input:
            image = dist.shard_tensor(image, dist_attr=dist.DistAttr(mesh=self._mesh, sharding_specs=['x', None]))
        out = layer(image)
        loss = loss_fn(out, label)
        loss.backward()
        grads = paddle.base.core.eager.get_grads_types([layer.parameters()[0], layer.parameters()[1]])
        layer.parameters()[0]._reset_grad_inplace_version()
        tmp = layer.parameters()[1]._grad_value()

    def test_demo_net(self):
        if False:
            return 10
        mp_layer = dist.shard_layer(DemoNet('grad_api_demo'), self._mesh, self.shard_fn)
        self.run_dynamic_grad_api(mp_layer)

    def run_test_case(self):
        if False:
            i = 10
            return i + 15
        self.test_demo_net()
if __name__ == '__main__':
    TestSimpleNetWithGradApiForSemiAutoParallel().run_test_case()
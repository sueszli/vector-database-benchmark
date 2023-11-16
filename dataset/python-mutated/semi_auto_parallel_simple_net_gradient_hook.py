import os
import numpy as np
from semi_auto_parallel_simple_net import DemoNet, TestSimpleNetForSemiAutoParallel
import paddle
import paddle.distributed as dist
from paddle import nn
hook_triggered = False

def backward_hook():
    if False:
        while True:
            i = 10

    def trigger_hook(grad):
        if False:
            return 10
        global hook_triggered
        hook_triggered = True
        assert grad.is_dist()
        return paddle.scale(grad, 1.0)
    return trigger_hook

class TestSimpleNetWithGradientHookForSemiAutoParallel(TestSimpleNetForSemiAutoParallel):

    def __init__(self):
        if False:
            print('Hello World!')
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        paddle.set_device(self._backend)

    def run_dynamic(self, layer):
        if False:
            return 10
        (image, label) = self.init_input_data()
        loss_fn = nn.MSELoss()
        out = layer(image)
        loss = loss_fn(out, label)
        loss.backward()

    def test_register_grad_hook(self):
        if False:
            i = 10
            return i + 15
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        model = dist.shard_layer(DemoNet('mp_demo_register_grad_hook'), self._mesh, self.shard_fn)
        model.parameters()[0]._register_grad_hook(backward_hook())
        self.run_dynamic(model)
        global hook_triggered
        assert hook_triggered
        hook_triggered = False

    def test_register_hook(self):
        if False:
            i = 10
            return i + 15
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        model = dist.shard_layer(DemoNet('mp_demo_register_hook'), self._mesh, self.shard_fn)
        model.parameters()[0].register_hook(backward_hook())
        self.run_dynamic(model)
        global hook_triggered
        assert hook_triggered
        hook_triggered = False

    def run_test_case(self):
        if False:
            i = 10
            return i + 15
        self.test_register_grad_hook()
        self.test_register_hook()
if __name__ == '__main__':
    TestSimpleNetWithGradientHookForSemiAutoParallel().run_test_case()
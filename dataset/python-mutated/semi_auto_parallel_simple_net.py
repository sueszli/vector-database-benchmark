import os
import random
import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.fleet.utils import recompute
BATCH_SIZE = 16
BATCH_NUM = 4
IMAGE_SIZE = 784
CLASS_NUM = 10

def create_numpy_like_random(name):
    if False:
        print('Hello World!')
    return paddle.ParamAttr(name=name, initializer=paddle.nn.initializer.Uniform(0, 1))

class DemoNet(nn.Layer):

    def __init__(self, param_prefix='', is_recompute=False, is_pp=False, pp_reshard_dist_attr=None):
        if False:
            return 10
        super().__init__()
        weight_attr_0 = create_numpy_like_random(param_prefix + '_0')
        weight_attr_1 = create_numpy_like_random(param_prefix + '_1')
        self.is_pp = is_pp
        self.is_recompute = is_recompute
        self.pp_reshard_dist_attr = pp_reshard_dist_attr
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, weight_attr_0)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, weight_attr_1)
        self.relu = nn.ReLU()

    def _inner_forward_fn(self, x):
        if False:
            print('Hello World!')
        out = self.linear_0(x)
        out = self.relu(out)
        if self.is_pp:
            out = dist.reshard(out, self.pp_reshard_dist_attr)
        out = self.linear_1(out)
        return out

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        if self.is_recompute:
            return recompute(self._inner_forward_fn, x)
        else:
            return self._inner_forward_fn(x)

class TestSimpleNetForSemiAutoParallel:

    def __init__(self):
        if False:
            return 10
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        self._pp_mesh0 = dist.ProcessMesh([0], dim_names=['x'])
        self._pp_mesh1 = dist.ProcessMesh([1], dim_names=['x'])
        self.pp_reshard_dist_attr = dist.DistAttr(mesh=self._pp_mesh1, sharding_specs=[None, None])
        paddle.set_device(self._backend)
        self.init_single_card_net_result()

    def shard_fn(self, layer_name, layer, process_mesh):
        if False:
            return 10
        if layer_name == 'linear_0':
            dist_attr = dist.DistAttr(mesh=process_mesh, sharding_specs=[None, 'x'])
            layer.weight = dist.shard_tensor(layer.weight, dist_attr=dist_attr)
        elif layer_name == 'linear_1':
            dist_attr = dist.DistAttr(mesh=process_mesh, sharding_specs=['x', None])
            layer.weight = dist.shard_tensor(layer.weight, dist_attr=dist_attr)

    def pp_shard_fn(self, layer_name, layer, process_mesh):
        if False:
            while True:
                i = 10
        if layer_name == 'linear_0':
            weight_dist_attr = dist.DistAttr(mesh=self._pp_mesh0, sharding_specs=[None, None])
            bias_dist_attr = dist.DistAttr(mesh=self._pp_mesh0, sharding_specs=[None])
            layer.weight = dist.shard_tensor(layer.weight, dist_attr=weight_dist_attr)
            layer.bias = dist.shard_tensor(layer.bias, dist_attr=bias_dist_attr)
        elif layer_name == 'linear_1':
            weight_dist_attr = dist.DistAttr(mesh=self._pp_mesh1, sharding_specs=[None, None])
            bias_dist_attr = dist.DistAttr(mesh=self._pp_mesh1, sharding_specs=[None])
            layer.weight = dist.shard_tensor(layer.weight, dist_attr=weight_dist_attr)
            layer.bias = dist.shard_tensor(layer.bias, dist_attr=bias_dist_attr)

    def set_random_seed(self, seed):
        if False:
            while True:
                i = 10
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def init_input_data(self):
        if False:
            for i in range(10):
                print('nop')
        image = np.random.random([BATCH_SIZE, IMAGE_SIZE]).astype('float32')
        label = np.random.random([BATCH_SIZE, CLASS_NUM]).astype('float32')
        return (paddle.to_tensor(image), paddle.to_tensor(label))

    def run_dynamic(self, layer, shard_input=False, is_pp=False):
        if False:
            i = 10
            return i + 15
        loss_fn = nn.MSELoss()
        input_mesh = self._pp_mesh0 if is_pp else self._mesh
        opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=layer.parameters())
        for _ in range(1):
            (image, label) = self.init_input_data()
            if shard_input:
                image = dist.shard_tensor(image, dist_attr=dist.DistAttr(mesh=input_mesh, sharding_specs=['x', None]))
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return (loss, layer.parameters())

    def init_single_card_net_result(self):
        if False:
            return 10
        self.set_random_seed(self._seed)
        (self.base_loss, self.base_parameters) = self.run_dynamic(DemoNet('demo_weight'))

    def check_tensor_eq(self, a, b, rtol=1e-05, atol=0, verbose=True):
        if False:
            return 10
        np1 = a.astype('float32').numpy()
        np2 = b.astype('float32').numpy()
        np.testing.assert_allclose(np1, np2, rtol=rtol, atol=atol, verbose=verbose)

    def test_dp_demo_net(self):
        if False:
            return 10
        self.set_random_seed(self._seed)
        (self.dp_loss, self.dp_parameters) = self.run_dynamic(DemoNet('dp_demo_weight'), shard_input=True)
        self.check_tensor_eq(self.dp_loss, self.base_loss)
        for (param, param_base) in zip(self.dp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def test_mp_demo_net(self):
        if False:
            print('Hello World!')
        self.set_random_seed(self._seed)
        mp_layer = dist.shard_layer(DemoNet('mp_demo_weight'), self._mesh, self.shard_fn)
        (self.mp_loss, self.mp_parameters) = self.run_dynamic(mp_layer)
        self.check_tensor_eq(self.mp_loss, self.base_loss)
        for (param, param_base) in zip(self.mp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def test_pp_demo_net(self):
        if False:
            print('Hello World!')
        self.set_random_seed(self._seed)
        if self._backend != 'gpu':
            return
        pp_layer = dist.shard_layer(DemoNet('pp_demo_weight', is_pp=True, pp_reshard_dist_attr=self.pp_reshard_dist_attr), self._pp_mesh0, self.pp_shard_fn)
        (self.pp_loss, self.pp_parameters) = self.run_dynamic(pp_layer, is_pp=True)
        rank = dist.get_rank()
        if rank == 0:
            self.check_tensor_eq(self.pp_parameters[0], self.base_parameters[0])
            self.check_tensor_eq(self.pp_parameters[1], self.base_parameters[1])
        else:
            self.check_tensor_eq(self.pp_loss, self.base_loss)
            self.check_tensor_eq(self.pp_parameters[2], self.base_parameters[2])
            self.check_tensor_eq(self.pp_parameters[3], self.base_parameters[3])

    def run_test_case(self):
        if False:
            print('Hello World!')
        self.test_dp_demo_net()
        self.test_mp_demo_net()
        self.test_pp_demo_net()
if __name__ == '__main__':
    TestSimpleNetForSemiAutoParallel().run_test_case()
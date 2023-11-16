import os
from site import getsitepackages
from semi_auto_parallel_simple_net import TestSimpleNetForSemiAutoParallel
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import IS_WINDOWS, run_cmd
paddle_includes = []
for site_packages_path in getsitepackages():
    paddle_includes.append(os.path.join(site_packages_path, 'paddle', 'include'))
    paddle_includes.append(os.path.join(site_packages_path, 'paddle', 'include', 'third_party'))
extra_cc_args = ['-w', '-g'] if not IS_WINDOWS else ['/w']
extra_nvcc_args = ['-O3']
file = f'{get_build_directory()}\\dist_custom_relu\\dist_custom_relu.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)
if os.name == 'nt':
    test_include = '..\\python\\paddle\\base\\tests\\auto_parallel'
else:
    test_include = '../python/paddle/base/tests/auto_parallel'
paddle_includes.append(test_include)
custom_ops = load(name='dist_custom_relu_jit', sources=['../custom_op/custom_relu_op.cc', '../custom_op/custom_relu_op_dup.cc', '../custom_op/custom_relu_op.cu'], extra_include_paths=paddle_includes, extra_cxx_cflags=extra_cc_args, extra_cuda_cflags=extra_nvcc_args, verbose=True)
BATCH_SIZE = 16
BATCH_NUM = 4
IMAGE_SIZE = 784
CLASS_NUM = 10

class PPDemoNet(nn.Layer):

    def __init__(self, mesh0, mesh1, param_suffix=''):
        if False:
            print('Hello World!')
        super().__init__()
        self.replicate_dist_attr0 = dist.DistAttr(mesh=mesh0, sharding_specs=[None, None])
        self.replicate_dist_attr1 = dist.DistAttr(mesh=mesh1, sharding_specs=[None, None])
        self.w0 = dist.shard_tensor(self.create_parameter(shape=[IMAGE_SIZE, IMAGE_SIZE], attr=paddle.framework.ParamAttr(name='pp_demo_weight_0' + param_suffix, initializer=paddle.nn.initializer.Uniform(0, 1))), dist_attr=self.replicate_dist_attr0)
        self.w1 = dist.shard_tensor(self.create_parameter(shape=[IMAGE_SIZE, CLASS_NUM], attr=paddle.framework.ParamAttr(name='pp_nemo_weight_1' + param_suffix, initializer=paddle.nn.initializer.Uniform(0, 1))), dist_attr=self.replicate_dist_attr1)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        out = F.linear(x, self.w0)
        out = custom_ops.custom_relu(out)
        out = dist.reshard(out, dist_attr=self.replicate_dist_attr1)
        out = F.linear(out, self.w1)
        return out

class TestSimpleNetWithCustomReluForSemiAutoParallel(TestSimpleNetForSemiAutoParallel):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        self._pp_mesh0 = dist.ProcessMesh([0], dim_names=['x'])
        self._pp_mesh1 = dist.ProcessMesh([1], dim_names=['x'])
        paddle.set_device(self._backend)

    def run_dynamic_custom_relu(self, layer, shard_input=False):
        if False:
            print('Hello World!')
        loss_fn = nn.MSELoss()
        (image, label) = self.init_input_data()
        if shard_input:
            image = dist.shard_tensor(image, dist_attr=dist.DistAttr(mesh=self._mesh, sharding_specs=['x', None]))
        out = layer(image)
        loss = loss_fn(out, label)
        loss.backward()

    def test_demo_net(self):
        if False:
            for i in range(10):
                print('nop')
        mp_layer = dist.shard_layer(PPDemoNet(self._pp_mesh0, self._pp_mesh1), self._mesh, self.shard_fn)
        self.run_dynamic_custom_relu(mp_layer)

    def run_test_case(self):
        if False:
            i = 10
            return i + 15
        self.test_demo_net()
if __name__ == '__main__':
    TestSimpleNetWithCustomReluForSemiAutoParallel().run_test_case()
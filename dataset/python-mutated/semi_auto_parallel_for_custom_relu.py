import os
from site import getsitepackages
import numpy as np
import paddle
import paddle.distributed as dist
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

class TestCustomReluForSemiAutoParallel:

    def __init__(self):
        if False:
            return 10
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])

    def check_tensor_eq(self, a, b):
        if False:
            return 10
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    def test_body(self, x_shape, x_specs):
        if False:
            for i in range(10):
                print('nop')
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        x_np = np.random.random(size=x_shape).astype(self._dtype)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        x_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=x_specs)
        dist_x = dist.shard_tensor(x_np, dist_attr=x_dist_attr)
        dist_x.stop_gradient = False
        y = paddle.add(x, x)
        dist_y = paddle.add(dist_x, dist_x)
        out = custom_ops.custom_relu(y)
        dist_out = custom_ops.custom_relu(dist_y)
        out.stop_gradient = False
        dist_out.stop_gradient = False
        self.check_tensor_eq(out, dist_out)
        out.backward()
        dist_out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad)

    def test_custom_relu(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_body(x_shape=[64, 32], x_specs=['x', None])

    def run_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.set_device('gpu:' + str(dist.get_rank()))
        self.test_custom_relu()
if __name__ == '__main__':
    TestCustomReluForSemiAutoParallel().test_custom_relu()
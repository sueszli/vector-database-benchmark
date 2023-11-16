import unittest
import numpy as np
import paddle
import paddle.distributed as dist

class TestSavedTensorHookForSemiAutoParallel(unittest.TestCase):

    def run_test_case(self):
        if False:
            while True:
                i = 10

        def pack_hook(x):
            if False:
                return 10
            return x.numpy()

        def unpack_hook(x):
            if False:
                for i in range(10):
                    print('nop')
            return paddle.to_tensor(x)
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        x_np = np.random.random(size=[64, 32]).astype(np.float32)
        y_np = np.random.random(size=[32, 48]).astype(np.float32)
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        x.stop_gradient = False
        y.stop_gradient = False
        x_dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None, None])
        y_dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None, None])
        dist_x = dist.shard_tensor(x_np, dist_attr=x_dist_attr)
        dist_y = dist.shard_tensor(y_np, dist_attr=y_dist_attr)
        dist_x.stop_gradient = False
        dist_y.stop_gradient = False
        with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
            z = paddle.matmul(x, y, False, False)
        with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
            dist_z = paddle.matmul(dist_x, dist_y, False, False)
        np.testing.assert_allclose(z.numpy(), dist_z.numpy(), rtol=0.0001, verbose=True)
        z.backward()
        dist_z.backward()
        np.testing.assert_allclose(x.grad.numpy(), dist_x.grad.numpy(), rtol=0.0001, verbose=True)
        np.testing.assert_allclose(y.grad.numpy(), dist_y.grad.numpy(), rtol=0.0001, verbose=True)
if __name__ == '__main__':
    TestSavedTensorHookForSemiAutoParallel().run_test_case()
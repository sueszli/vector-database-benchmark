import unittest
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.autograd.py_layer import PyLayer

class TestNet(PyLayer):

    @staticmethod
    def forward(ctx, x1, x2, x3):
        if False:
            i = 10
            return i + 15
        y1 = paddle.matmul(x1, x2, transpose_x=False, transpose_y=False)
        y2 = paddle.matmul(x2, x3, transpose_x=False, transpose_y=False)
        return (y1, y2)

    @staticmethod
    def backward(ctx, dy1, dy2):
        if False:
            i = 10
            return i + 15
        return (dy1, dy2, dy2)

class TestPyLayerForSemiAutoParallel(unittest.TestCase):

    def run_test_case(self):
        if False:
            print('Hello World!')
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        x1_np = np.random.random(size=[64, 32]).astype(np.float32)
        x2_np = np.random.random(size=[32, 48]).astype(np.float32)
        x3_np = np.random.random(size=[48, 64]).astype(np.float32)
        x1 = paddle.to_tensor(x1_np)
        x2 = paddle.to_tensor(x2_np)
        x3 = paddle.to_tensor(x3_np)
        x1.stop_gradient = False
        x2.stop_gradient = False
        x3.stop_gradient = False
        x1_dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None, None])
        x2_dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None, None])
        x3_dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None, None])
        dist_x1 = dist.shard_tensor(x1_np, dist_attr=x1_dist_attr)
        dist_x2 = dist.shard_tensor(x2_np, dist_attr=x2_dist_attr)
        dist_x3 = dist.shard_tensor(x3_np, dist_attr=x3_dist_attr)
        dist_x1.stop_gradient = False
        dist_x2.stop_gradient = False
        dist_x3.stop_gradient = False
        (y1, y2) = TestNet.apply(x1, x2, x3)
        loss = y1.sum()
        (dist_y1, dist_y2) = TestNet.apply(dist_x1, dist_x2, dist_x3)
        dist_loss = dist_y1.sum()
        np.testing.assert_allclose(loss.numpy(), dist_loss.numpy(), rtol=0.0001, verbose=True)
        loss.backward()
        dist_loss.backward()
        np.testing.assert_allclose(x1.grad.numpy(), dist_x1.grad.numpy(), rtol=0.0001, verbose=True)
        np.testing.assert_allclose(x2.grad.numpy(), dist_x2.grad.numpy(), rtol=0.0001, verbose=True)
        np.testing.assert_allclose(x3.grad.numpy(), dist_x3.grad.numpy(), rtol=0.0001, verbose=True)
if __name__ == '__main__':
    TestPyLayerForSemiAutoParallel().run_test_case()
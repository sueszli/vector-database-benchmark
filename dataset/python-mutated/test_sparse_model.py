import unittest
import numpy as np
import paddle
from paddle.sparse import nn

class TestGradientAdd(unittest.TestCase):

    def sparse(self, sp_x):
        if False:
            for i in range(10):
                print('nop')
        indentity = sp_x
        out = nn.functional.relu(sp_x)
        values = out.values() + indentity.values()
        out = paddle.sparse.sparse_coo_tensor(out.indices(), values, shape=out.shape, stop_gradient=out.stop_gradient)
        return out

    def dense(self, x):
        if False:
            i = 10
            return i + 15
        indentity = x
        out = paddle.nn.functional.relu(x)
        out = out + indentity
        return out

    def test(self):
        if False:
            return 10
        x = paddle.randn((3, 3))
        sparse_x = x.to_sparse_coo(sparse_dim=2)
        x.stop_gradient = False
        sparse_x.stop_gradient = False
        dense_out = self.dense(x)
        loss = dense_out.mean()
        loss.backward(retain_graph=True)
        sparse_out = self.sparse(sparse_x)
        sparse_loss = sparse_out.values().mean()
        sparse_loss.backward(retain_graph=True)
        np.testing.assert_allclose(dense_out.numpy(), sparse_out.to_dense().numpy())
        np.testing.assert_allclose(x.grad.numpy(), sparse_x.grad.to_dense().numpy())
        loss.backward()
        sparse_loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), sparse_x.grad.to_dense().numpy())
if __name__ == '__main__':
    unittest.main()
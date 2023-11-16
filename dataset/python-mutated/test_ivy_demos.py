"""Collection of tests for the demos."""
import pytest
import ivy
import ivy.functional.backends.numpy

def test_array(on_device):
    if False:
        return 10
    import jax.numpy as jnp
    assert ivy.concat((jnp.ones((1,)), jnp.ones((1,))), axis=-1).shape == (2,)
    import tensorflow as tf
    assert ivy.concat((tf.ones((1,)), tf.ones((1,))), axis=-1).shape == (2,)
    import numpy as np
    assert ivy.concat((np.ones((1,)), np.ones((1,))), axis=-1).shape == (2,)
    import torch
    assert ivy.concat((torch.ones((1,)), torch.ones((1,))), axis=-1).shape == (2,)
    import paddle
    assert ivy.concat((paddle.ones((1,)), paddle.ones((1,))), axis=-1).shape == (2,)

def test_training_demo(on_device, backend_fw):
    if False:
        print('Hello World!')
    if backend_fw == 'numpy':
        pytest.skip()
    ivy.set_backend(backend_fw)

    class MyModel(ivy.Module):

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.linear0 = ivy.Linear(3, 64)
            self.linear1 = ivy.Linear(64, 1)
            ivy.Module.__init__(self)

        def _forward(self, x):
            if False:
                for i in range(10):
                    print('nop')
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear1(x))
    model = MyModel()
    optimizer = ivy.Adam(0.0001)
    x_in = ivy.array([1.0, 2.0, 3.0])
    target = ivy.array([0.0])

    def loss_fn(v):
        if False:
            i = 10
            return i + 15
        out = model(x_in, v=v)
        return ivy.mean((out - target) ** 2)
    for step in range(100):
        (loss, grads) = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)
    ivy.previous_backend()
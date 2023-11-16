import unittest
from time import time
import numpy as np
from dygraph_to_static_utils_new import test_legacy_and_pir
from test_mnist import MNIST, SEED, TestMNIST
import paddle
from paddle.optimizer import Adam
if paddle.base.is_compiled_with_cuda():
    paddle.base.set_flags({'FLAGS_cudnn_deterministic': True})

class TestAMP(TestMNIST):

    def train_static(self):
        if False:
            print('Hello World!')
        return self.train(to_static=True)

    def train_dygraph(self):
        if False:
            while True:
                i = 10
        return self.train(to_static=False)

    @test_legacy_and_pir
    def test_mnist_to_static(self):
        if False:
            return 10
        dygraph_loss = self.train_dygraph()
        static_loss = self.train_static()
        np.testing.assert_allclose(dygraph_loss, static_loss, rtol=1e-05, atol=0.001, err_msg=f'dygraph is {dygraph_loss}\n static_res is \n{static_loss}')

    def train(self, to_static=False):
        if False:
            return 10
        paddle.seed(SEED)
        mnist = MNIST()
        if to_static:
            print('Successfully to apply @to_static.')
            mnist = paddle.jit.to_static(mnist)
        adam = Adam(learning_rate=0.001, parameters=mnist.parameters())
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        loss_data = []
        for epoch in range(self.epoch_num):
            start = time()
            for (batch_id, data) in enumerate(self.train_reader()):
                dy_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
                img = paddle.to_tensor(dy_x_data)
                label = paddle.to_tensor(y_data)
                label.stop_gradient = True
                with paddle.amp.auto_cast():
                    (prediction, acc, avg_loss) = mnist(img, label=label)
                scaled = scaler.scale(avg_loss)
                scaled.backward()
                scaler.minimize(adam, scaled)
                loss_data.append(float(avg_loss))
                mnist.clear_gradients()
                if batch_id % 10 == 0:
                    print('Loss at epoch {} step {}: loss: {:}, acc: {}, cost: {}'.format(epoch, batch_id, avg_loss.numpy(), acc.numpy(), time() - start))
                    start = time()
                if batch_id == 50:
                    break
        return loss_data
if __name__ == '__main__':
    unittest.main()
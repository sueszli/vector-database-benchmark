from dist_mnist import cnn_model
from test_dist_base import TestDistRunnerBase, runtime_main
import paddle
from paddle import base
DTYPE = 'float32'
paddle.dataset.mnist.fetch()
base.default_startup_program().random_seed = 1
base.default_main_program().random_seed = 1

class TestDistMnist2x2(TestDistRunnerBase):

    def get_model(self, batch_size=2):
        if False:
            i = 10
            return i + 15
        images = paddle.static.data(name='pixel', shape=[-1, 1, 28, 28], dtype=DTYPE)
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        predict = cnn_model(images)
        cost = paddle.nn.functional.cross_entropy(input=predict, label=label, reduction='none', use_softmax=False)
        avg_cost = paddle.mean(x=cost)
        batch_size_tensor = paddle.tensor.create_tensor(dtype='int64')
        batch_acc = paddle.static.accuracy(input=predict, label=label, total=batch_size_tensor)
        inference_program = base.default_main_program().clone()
        opt = paddle.incubate.optimizer.LarsMomentumOptimizer(learning_rate=0.001, momentum=0.9)
        train_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=batch_size)
        test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=batch_size)
        opt.minimize(avg_cost)
        return (inference_program, avg_cost, train_reader, test_reader, batch_acc, predict)
if __name__ == '__main__':
    runtime_main(TestDistMnist2x2)
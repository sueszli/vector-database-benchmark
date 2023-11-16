from dist_mnist import cnn_model
from test_dist_base import TestDistRunnerBase, runtime_main
import paddle
from paddle import base
DTYPE = 'float32'

def test_merge_reader(repeat_batch_size=8):
    if False:
        return 10
    orig_reader = paddle.dataset.mnist.test()
    record_batch = []
    b = 0
    for d in orig_reader():
        if b >= repeat_batch_size:
            break
        record_batch.append(d)
        b += 1
    while True:
        for d in record_batch:
            yield d

class TestDistMnist2x2(TestDistRunnerBase):

    def get_model(self, batch_size=2):
        if False:
            while True:
                i = 10
        images = paddle.static.data(name='pixel', shape=[-1, 1, 28, 28], dtype=DTYPE)
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        predict = cnn_model(images)
        cost = paddle.nn.functional.cross_entropy(input=predict, label=label, reduction='none', use_softmax=False)
        avg_cost = paddle.mean(x=cost)
        batch_size_tensor = paddle.tensor.create_tensor(dtype='int64')
        batch_acc = paddle.static.accuracy(input=predict, label=label, total=batch_size_tensor)
        inference_program = base.default_main_program().clone()
        opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        train_reader = paddle.batch(test_merge_reader, batch_size=batch_size)
        test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=batch_size)
        opt.minimize(avg_cost)
        return (inference_program, avg_cost, train_reader, test_reader, batch_acc, predict)
if __name__ == '__main__':
    runtime_main(TestDistMnist2x2)
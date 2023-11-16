import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_training(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_training = True
        self.epoch = 1

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        self.batch_size = 16
        self.max_seq_length = 5
        self.max_label_length = 3
        self.num_classes = 5
        self.logits_length = np.array([self.max_seq_length] * self.batch_size, dtype=np.int64)
        self.labels_length = np.array([self.max_label_length] * self.batch_size, dtype=np.int64)
        self.blank = self.num_classes - 1
        self.norm_by_times = False
        logits = np.random.uniform(0.1, 1.0, [self.max_seq_length, self.batch_size, self.num_classes]).astype('float32')
        labels = np.random.randint(0, self.num_classes - 1, [self.batch_size, self.max_label_length], dtype='int32')
        self.feed_fp32 = {'Logits': logits, 'Label': labels, 'input_length': self.logits_length.astype('int64'), 'label_length': self.labels_length.astype('int64')}
        self.feed_fp16 = {'Logits': logits.astype(np.float16), 'Label': labels, 'input_length': self.logits_length.astype('int64'), 'label_length': self.labels_length.astype('int64')}

    def set_feed_attr(self):
        if False:
            print('Hello World!')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'blank': self.blank, 'norm_by_times': self.norm_by_times}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            i = 10
            return i + 15
        data = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        logits = paddle.nn.Linear(self.num_classes, self.num_classes, bias_attr=False)(data)
        labels = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='int32')
        input_length = paddle.static.data(name=self.feed_list[2], shape=self.feed_shape[2], dtype='int64')
        label_length = paddle.static.data(name=self.feed_list[3], shape=self.feed_shape[3], dtype='int64')
        out = paddle.nn.functional.ctc_loss(logits, labels, input_length=input_length, label_length=label_length, reduction='mean', **self.attrs)
        loss = paddle.mean(out)
        adam = paddle.optimizer.Adam(learning_rate=0.01)
        adam.minimize(loss)
        self.fetch_list = [loss.name, out.name]

    def run_model(self, exec_mode):
        if False:
            return 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            return 10
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()
if __name__ == '__main__':
    unittest.main()
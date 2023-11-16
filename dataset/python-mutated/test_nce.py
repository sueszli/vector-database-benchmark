import unittest
import numpy as np
from op_test import OpTest, paddle_static_guard
import paddle
from paddle import base
from paddle.base import Program, program_guard

def nce(input, weight, bias, sample_weight, labels, num_classes, num_sample_class):
    if False:
        print('Hello World!')
    samples = []
    sample_labels = []
    batch_size = input.shape[0]
    num_true_class = labels.shape[1]
    for i in range(batch_size):
        w = 1 if sample_weight is None else sample_weight[i]
        for label in labels[i]:
            samples.append((i, label, True, w))
            sample_labels.append(label)
        for num in range(num_sample_class):
            samples.append((i, num, False, w))
            sample_labels.append(num)
    sample_out = np.zeros(len(samples)).astype(np.float32)
    if bias is not None:
        for i in range(len(samples)):
            sample_out[i] = bias[samples[i][1]]
    for i in range(len(samples)):
        sample_out[i] += np.dot(input[samples[i][0]], weight[samples[i][1]])
    sample_out = 1.0 / (1.0 + np.exp(-sample_out))
    out = np.zeros(batch_size).astype(np.float32)
    b = 1.0 / num_classes * num_sample_class
    for i in range(len(samples)):
        o = sample_out[i]
        cost = -np.log(o / (o + b)) if samples[i][2] else -np.log(b / (o + b))
        out[samples[i][0]] += cost * samples[i][3]
    return (out[:, np.newaxis], np.array(sample_out).reshape(batch_size, num_sample_class + num_true_class), np.array(sample_labels).reshape(batch_size, num_sample_class + num_true_class))

class TestNCE(OpTest):

    def generate_data(self, dim, batch_size, num_classes, num_true_class, num_neg_samples, is_sparse):
        if False:
            print('Hello World!')
        input = np.random.randn(batch_size, dim).astype(np.float32)
        weight = np.random.randn(num_classes, dim).astype(np.float32)
        bias = np.random.randn(num_classes).astype(np.float32)
        sample_weight = np.random.randn(batch_size).astype(np.float32)
        labels = np.random.randint(0, num_classes, (batch_size, num_true_class)).astype('int64')
        self.attrs = {'num_total_classes': num_classes, 'num_neg_samples': num_neg_samples, 'custom_neg_classes': list(range(num_neg_samples)), 'seed': 0, 'sampler': 0, 'is_sparse': is_sparse, 'is_test': self.is_test}
        self.inputs = {'Input': input, 'Label': labels, 'Weight': weight, 'Bias': bias, 'SampleWeight': sample_weight}

    def set_is_test(self):
        if False:
            print('Hello World!')
        self.is_test = False

    def set_data(self):
        if False:
            return 10
        self.generate_data(5, 25, 100, 1, 2, False)

    def compute(self):
        if False:
            i = 10
            return i + 15
        out = nce(self.inputs['Input'], self.inputs['Weight'], self.inputs['Bias'], self.inputs['SampleWeight'], self.inputs['Label'], self.attrs['num_total_classes'], self.attrs['num_neg_samples'])
        if self.is_test:
            self.outputs = {'Cost': out[0]}
        else:
            self.outputs = {'Cost': out[0], 'SampleLogits': out[1], 'SampleLabels': out[2]}

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'nce'
        self.set_is_test()
        self.set_data()
        self.compute()

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output()

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['Input', 'Weight', 'Bias'], 'Cost', max_relative_error=0.02)

class TestNCECase1Tensor(TestNCE):

    def set_data(self):
        if False:
            while True:
                i = 10
        self.generate_data(10, 20, 100, 2, 5, False)

class TestNCETensorIsTest(TestNCE):

    def set_is_test(self):
        if False:
            while True:
                i = 10
        self.is_test = True

    def test_check_grad(self):
        if False:
            print('Hello World!')
        pass

class TestNCECase1SelectedRows(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.base_lr = 0.0001
        self.batch_size = 8

    @staticmethod
    def get_place():
        if False:
            i = 10
            return i + 15
        place = base.core.CPUPlace()
        return place

    @staticmethod
    def get_train_data(batch_size):
        if False:
            return 10
        batches = []
        for i in range(batch_size):
            input = np.random.randn(batch_size, 10).astype(np.float32)
            labels = np.random.randint(0, 20, (batch_size, 1))
            batches.append([input, labels])
        return batches

    def get_optimizer(self):
        if False:
            while True:
                i = 10
        optimizer = paddle.optimizer.SGD(learning_rate=self.base_lr)
        return optimizer

    def train_network(self, num_total_classes, num_neg_samples, sampler, custom_dist, is_sparse):
        if False:
            return 10
        with paddle_static_guard():
            input = paddle.static.data(name='input', shape=[-1, 10], dtype='float32')
            label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
            w_param = base.default_main_program().global_block().create_parameter(shape=[num_total_classes, 10], dtype='float32', name='nce_w', initializer=paddle.nn.initializer.Constant())
            b_param = base.default_main_program().global_block().create_parameter(shape=[num_total_classes, 1], dtype='float32', name='nce_b', initializer=paddle.nn.initializer.Constant())
            cost = paddle.static.nn.nce(input=input, label=label, num_total_classes=num_total_classes, sampler=sampler, custom_dist=custom_dist, sample_weight=None, param_attr='nce_w', bias_attr='nce_b', seed=1, num_neg_samples=num_neg_samples, is_sparse=is_sparse)
            avg_cost = paddle.mean(cost)
            optimizer = self.get_optimizer()
            optimizer.minimize(avg_cost)
            return [avg_cost, [input, label]]

    def test_input_is_selected_rows(self):
        if False:
            print('Hello World!')
        with paddle_static_guard():
            place = self.get_place()
            exe = base.Executor(place)
            data = self.get_train_data(self.batch_size)
            nid_freq_arr = np.random.dirichlet(np.ones(20) * 1000).astype('float32')
            rets = []
            dense_scope = base.core.Scope()
            dense_startup_program = base.framework.Program()
            dense_train_program = base.framework.Program()
            with base.scope_guard(dense_scope):
                with base.program_guard(dense_train_program, dense_startup_program):
                    (cost, feeds) = self.train_network(20, 5, 'custom_dist', nid_freq_arr.tolist(), False)
                    feeder = base.DataFeeder(feed_list=feeds, place=place)
                    paddle.enable_static()
                    exe.run(dense_startup_program)
                    loss_val = exe.run(dense_train_program, feed=feeder.feed(data), fetch_list=[cost.name])
                    rets.append(np.mean(loss_val))
            sparse_scope = base.core.Scope()
            sparse_startup_program = base.framework.Program()
            sparse_train_program = base.framework.Program()
            with base.scope_guard(sparse_scope):
                with base.program_guard(sparse_train_program, sparse_startup_program):
                    (cost, feeds) = self.train_network(20, 5, 'custom_dist', nid_freq_arr.tolist(), True)
                    feeder = base.DataFeeder(feed_list=feeds, place=place)
                    paddle.enable_static()
                    exe.run(sparse_startup_program)
                    loss_val = exe.run(sparse_train_program, feed=feeder.feed(data), fetch_list=[cost.name])
                    rets.append(np.mean(loss_val))
            self.assertEqual(rets[0], rets[1])

class TestNCE_OpError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with paddle_static_guard():
            with program_guard(Program(), Program()):
                input1 = base.create_lod_tensor(np.array([0.0, 3.0, 2.0, 4.0]), [[1, 1, 2]], base.CPUPlace())
                label1 = paddle.static.data(name='label1', shape=[-1, 4], dtype='int64')
                self.assertRaises(TypeError, paddle.static.nn.nce, input1, label1, 5)
                input2 = paddle.static.data(name='input2', shape=[-1, 4], dtype='float32')
                label2 = base.create_lod_tensor(np.array([0.0, 3.0, 2.0, 4.0]), [[1, 1, 2]], base.CPUPlace())
                self.assertRaises(TypeError, paddle.static.nn.nce, input2, label2, 5)
                input3 = paddle.static.data(name='input3', shape=[-1, 4], dtype='float16')
                label3 = paddle.static.data(name='label3', shape=[-1, 1], dtype='int64')
                self.assertRaises(TypeError, paddle.static.nn.nce, input3, label3, 5)
                input4 = paddle.static.data(name='input4', shape=[-1, 4], dtype='float32')
                label4 = paddle.static.data(name='label4', shape=[-1, 1], dtype='int32')
                self.assertRaises(TypeError, paddle.static.nn.nce, input4, label4, 5)
                input5 = paddle.static.data(name='x', shape=[1], dtype='float32')
                label5 = paddle.static.data(name='label', shape=[1], dtype='int64')
                self.assertRaises(ValueError, paddle.static.nn.nce, input5, label5, 1)
if __name__ == '__main__':
    unittest.main()
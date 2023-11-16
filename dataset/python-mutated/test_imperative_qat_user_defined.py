import logging
import os
import unittest
import numpy as np
import paddle
from paddle import nn
from paddle.nn import Sequential
from paddle.optimizer import Adam
from paddle.quantization import ImperativeQuantAware
from paddle.static.log_helper import get_logger
os.environ['CPU_NUM'] = '1'
_logger = get_logger(__name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

class PACT(nn.Layer):

    def __init__(self, init_value=20):
        if False:
            return 10
        super().__init__()
        alpha_attr = paddle.ParamAttr(name=self.full_name() + '.pact', initializer=paddle.nn.initializer.Constant(value=init_value))
        self.alpha = self.create_parameter(shape=[1], attr=alpha_attr, dtype='float32')

    def forward(self, x):
        if False:
            return 10
        out_left = paddle.nn.functional.relu(x - self.alpha)
        out_right = paddle.nn.functional.relu(-self.alpha - x)
        x = x - out_left + out_right
        return x

class CustomQAT(nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0))
        self.u_param = self.create_parameter(shape=[1], attr=attr, dtype='float32')
        self.l_param = self.create_parameter(shape=[1], attr=attr, dtype='float32')
        self.alpha_param = self.create_parameter(shape=[1], attr=attr, dtype='float32')
        self.upper = self.create_parameter(shape=[1], attr=attr, dtype='float32')
        self.upper.stop_gradient = True
        self.lower = self.create_parameter(shape=[1], attr=attr, dtype='float32')
        self.lower.stop_gradient = True

    def forward(self, x):
        if False:
            print('Hello World!')

        def clip(x, upper, lower):
            if False:
                i = 10
                return i + 15
            x = x + paddle.nn.functional.relu(lower - x)
            x = x - paddle.nn.functional.relu(x - upper)
            return x

        def phi_function(x, mi, alpha, delta):
            if False:
                while True:
                    i = 10
            s = 1 / (1 - alpha)
            k = paddle.log(2 / alpha - 1) * (1 / delta)
            x = paddle.tanh((x - mi) * k) * s
            return x

        def dequantize(x, lower_bound, delta, interval):
            if False:
                print('Hello World!')
            x = ((x + 1) / 2 + interval) * delta + lower_bound
            return x
        bit = 8
        bit_range = 2 ** bit - 1
        paddle.assign(self.upper * 0.9 + self.u_param * 0.1, self.upper)
        paddle.assign(self.lower * 0.9 + self.l_param * 0.1, self.lower)
        x = clip(x, self.upper, self.lower)
        delta = (self.upper - self.lower) / bit_range
        interval = (x - self.lower) / delta
        mi = (interval + 0.5) * delta + self.l_param
        x = phi_function(x, mi, self.alpha_param, delta)
        x = dequantize(x, self.l_param, delta, interval)
        return x

class ModelForConv2dT(nn.Layer):

    def __init__(self, num_classes=10):
        if False:
            while True:
                i = 10
        super().__init__()
        self.features = nn.Conv2DTranspose(4, 6, (3, 3))
        self.fc = nn.Linear(in_features=600, out_features=num_classes)

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        x = self.features(inputs)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x

class ImperativeLenet(paddle.nn.Layer):

    def __init__(self, num_classes=10, classifier_activation='softmax'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.features = Sequential(nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1), nn.MaxPool2D(kernel_size=2, stride=2), nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0), nn.MaxPool2D(kernel_size=2, stride=2))
        self.fc = Sequential(nn.Linear(in_features=400, out_features=120), nn.Linear(in_features=120, out_features=84), nn.Linear(in_features=84, out_features=num_classes))

    def forward(self, inputs):
        if False:
            return 10
        x = self.features(inputs)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x

class TestUserDefinedActPreprocess(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        _logger.info('test act_preprocess')
        self.imperative_qat = ImperativeQuantAware(act_preprocess_layer=PACT)

    def func_quant_aware_training(self):
        if False:
            i = 10
            return i + 15
        imperative_qat = self.imperative_qat
        seed = 1
        np.random.seed(seed)
        paddle.static.default_main_program().random_seed = seed
        paddle.static.default_startup_program().random_seed = seed
        lenet = ImperativeLenet()
        fixed_state = {}
        param_init_map = {}
        for (name, param) in lenet.named_parameters():
            p_shape = np.array(param).shape
            p_value = np.array(param)
            if name.endswith('bias'):
                value = np.zeros_like(p_value).astype('float32')
            else:
                value = np.random.normal(loc=0.0, scale=0.01, size=np.prod(p_shape)).reshape(p_shape).astype('float32')
            fixed_state[name] = value
            param_init_map[param.name] = value
        lenet.set_dict(fixed_state)
        imperative_qat.quantize(lenet)
        adam = Adam(learning_rate=0.001, parameters=lenet.parameters())
        dynamic_loss_rec = []
        conv_transpose = ModelForConv2dT()
        imperative_qat.quantize(conv_transpose)
        x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1.0, max=1.0)
        conv_transpose(x_var)

        def train(model):
            if False:
                print('Hello World!')
            adam = Adam(learning_rate=0.001, parameters=model.parameters())
            epoch_num = 1
            for epoch in range(epoch_num):
                model.train()
                for (batch_id, data) in enumerate(train_reader()):
                    x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                    y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
                    img = paddle.to_tensor(x_data)
                    label = paddle.to_tensor(y_data)
                    out = model(img)
                    acc = paddle.metric.accuracy(out, label, k=1)
                    loss = nn.functional.loss.cross_entropy(out, label)
                    avg_loss = paddle.mean(loss)
                    avg_loss.backward()
                    adam.step()
                    adam.clear_grad()
                    if batch_id % 50 == 0:
                        _logger.info('Train | At epoch {} step {}: loss = {:}, acc= {:}'.format(epoch, batch_id, avg_loss.numpy(), acc.numpy()))
                        break

        def test(model):
            if False:
                i = 10
                return i + 15
            model.eval()
            avg_acc = [[], []]
            for (batch_id, data) in enumerate(test_reader()):
                x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
                img = paddle.to_tensor(x_data)
                label = paddle.to_tensor(y_data)
                out = model(img)
                acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
                acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
                avg_acc[0].append(acc_top1.numpy())
                avg_acc[1].append(acc_top5.numpy())
                if batch_id % 100 == 0:
                    _logger.info('Test | step {}: acc1 = {:}, acc5 = {:}'.format(batch_id, acc_top1.numpy(), acc_top5.numpy()))
        train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=512, drop_last=True)
        test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=512)
        train(lenet)
        test(lenet)

    def test_quant_aware_training(self):
        if False:
            return 10
        self.func_quant_aware_training()

class TestUserDefinedWeightPreprocess(TestUserDefinedActPreprocess):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        _logger.info('test weight_preprocess')
        self.imperative_qat = ImperativeQuantAware(weight_preprocess_layer=PACT)

class TestUserDefinedActQuantize(TestUserDefinedActPreprocess):

    def setUp(self):
        if False:
            while True:
                i = 10
        _logger.info('test act_quantize')
        self.imperative_qat = ImperativeQuantAware(act_quantize_layer=CustomQAT)

class TestUserDefinedWeightQuantize(TestUserDefinedActPreprocess):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        _logger.info('test weight_quantize')
        self.imperative_qat = ImperativeQuantAware(weight_quantize_layer=CustomQAT)
if __name__ == '__main__':
    unittest.main()
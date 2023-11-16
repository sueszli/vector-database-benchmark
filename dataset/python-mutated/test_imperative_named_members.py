import unittest
import numpy as np
import paddle
from paddle import base

class MyLayer(paddle.nn.Layer):

    def __init__(self, num_channel, dim, num_filter=5):
        if False:
            while True:
                i = 10
        super().__init__()
        self.fc = paddle.nn.Linear(dim, dim)
        self.conv = paddle.nn.Conv2D(num_channel, num_channel, num_filter)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.fc(x)
        x = self.conv(x)
        return x

class TestImperativeNamedSubLayers(unittest.TestCase):

    def test_named_sublayers(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            fc1 = paddle.nn.Linear(10, 3)
            fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
            custom = MyLayer(3, 10)
            model = paddle.nn.Sequential(fc1, fc2, custom)
            named_sublayers = model.named_sublayers()
            list_named_sublayers = list(named_sublayers)
            expected_sublayers = [fc1, fc2, custom, custom.fc, custom.conv]
            self.assertEqual(len(list_named_sublayers), len(expected_sublayers))
            for ((name, sublayer), expected_sublayer) in zip(list_named_sublayers, expected_sublayers):
                self.assertEqual(sublayer, expected_sublayer)
            list_sublayers = list(model.sublayers())
            self.assertEqual(len(list_named_sublayers), len(list_sublayers))
            for ((name, sublayer), expected_sublayer) in zip(list_named_sublayers, list_sublayers):
                self.assertEqual(sublayer, expected_sublayer)
            self.assertListEqual([l for (_, l) in list(model.named_sublayers(include_self=True))], [model] + expected_sublayers)

class TestImperativeNamedParameters(unittest.TestCase):

    def test_named_parameters(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard():
            fc1 = paddle.nn.Linear(10, 3)
            fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
            custom = MyLayer(3, 10)
            model = paddle.nn.Sequential(fc1, fc2, custom)
            named_parameters = list(model.named_parameters())
            expected_named_parameters = []
            for (prefix, layer) in model.named_sublayers():
                for (name, param) in layer.named_parameters(include_sublayers=False):
                    full_name = prefix + ('.' if prefix else '') + name
                    expected_named_parameters.append((full_name, param))
            self.assertListEqual(expected_named_parameters, named_parameters)

    def test_dir_layer(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard():

            class Mymodel(paddle.nn.Layer):

                def __init__(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    super().__init__()
                    self.linear1 = paddle.nn.Linear(10, 10)
                    self.linear2 = paddle.nn.Linear(5, 5)
                    self.conv2d = paddle.nn.Conv2D(3, 2, 3)
                    self.embedding = paddle.nn.Embedding(128, 16)
                    self.h_0 = base.dygraph.to_variable(np.zeros([10, 10]).astype('float32'))
                    self.weight = self.create_parameter(shape=[2, 3], attr=base.ParamAttr(), dtype='float32', is_bias=False)
            model = Mymodel()
            expected_members = dir(model)
            self.assertTrue('linear1' in expected_members, 'model should contain Layer: linear1')
            self.assertTrue('linear2' in expected_members, 'model should contain Layer: linear2')
            self.assertTrue('conv2d' in expected_members, 'model should contain Layer: conv2d')
            self.assertTrue('embedding' in expected_members, 'model should contain Layer: embedding')
            self.assertTrue('h_0' in expected_members, 'model should contain buffer: h_0')
            self.assertTrue('weight' in expected_members, 'model should contain parameter: weight')
if __name__ == '__main__':
    unittest.main()
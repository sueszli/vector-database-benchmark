import json
import os
import tempfile
import unittest
import warnings
import paddle
import paddle.nn.functional as F

class SimpleNet(paddle.nn.Layer):

    def __init__(self, data_format='NCHW', class_num=2):
        if False:
            return 10
        super().__init__()
        self.conv = paddle.nn.Conv2D(3, 8, (3, 3))
        self.bn = paddle.nn.BatchNorm(num_channels=8)
        self.relu = paddle.nn.ReLU()
        self.pool = paddle.nn.AvgPool2D(kernel_size=2, stride=2)
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(392, class_num)

    def forward(self, image):
        if False:
            print('Hello World!')
        conv_out = self.conv(image)
        bn_out = self.bn(conv_out)
        out = self.relu(bn_out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return (conv_out, out)

class LayoutAutoTune(unittest.TestCase):

    def test_config(self):
        if False:
            i = 10
            return i + 15
        paddle.base.core.enable_layout_autotune()
        if self.use_autoune():
            self.assertEqual(paddle.base.core.use_layout_autotune(), True)
            paddle.base.core.disable_layout_autotune()
        self.assertEqual(paddle.base.core.use_layout_autotune(), False)
        self.use_autoune()

    def setUp(self):
        if False:
            return 10
        self.use_autoune()

    def use_autoune(self):
        if False:
            while True:
                i = 10
        if paddle.is_compiled_with_cuda():
            paddle.incubate.autotune.set_config(config={'layout': {'enable': True}})
            return paddle.base.core.use_layout_autotune()
        else:
            config = {'layout': {'enable': False}}
            tfile = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            json.dump(config, tfile)
            tfile.close()
            paddle.incubate.autotune.set_config(tfile.name)
            os.remove(tfile.name)
            return paddle.base.core.use_layout_autotune()

    def train(self, data_format):
        if False:
            for i in range(10):
                print('nop')
        model = SimpleNet(data_format='NCHW', class_num=2)
        data = paddle.rand([1, 3, 16, 16])
        if data_format == 'NHWC':
            data = paddle.rand([1, 16, 16, 3])
        label_data = paddle.randint(0, 1, shape=[1, 1], dtype='int64')
        optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())
        scaler = paddle.amp.GradScaler()
        for i in range(2):
            with paddle.amp.auto_cast(level='O2'):
                (conv_out, predict) = model(data)
                loss = F.cross_entropy(predict, label=label_data)
                loss = loss.mean()
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
        return (conv_out, predict)

    def test_enable_autotune(self):
        if False:
            i = 10
            return i + 15
        (conv_out, predict) = self.train(data_format='NCHW')
        self.assertEqual(conv_out.shape, [1, 8, 14, 14])
        self.assertEqual(predict.shape, [1, 2])

    def test_transpose_op_transposer(self):
        if False:
            i = 10
            return i + 15
        conv = paddle.nn.Conv2D(3, 8, (3, 3))
        data = paddle.rand([1, 3, 16, 14])
        label_data = paddle.randint(0, 1, shape=[1, 1], dtype='int64')
        optimizer = paddle.optimizer.SGD(learning_rate=0.0001, parameters=conv.parameters())
        scaler = paddle.amp.GradScaler()
        with paddle.amp.auto_cast(level='O2'):
            conv_out = conv(data)
            out = paddle.transpose(conv_out, perm=[0, 3, 1, 2])
            loss = out.mean()
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(optimizer, scaled)
        self.assertEqual(conv_out.shape, [1, 8, 14, 12])
        self.assertEqual(out.shape, [1, 12, 8, 14])

    def test_flatten_op_transposer(self):
        if False:
            return 10
        conv = paddle.nn.Conv2D(3, 8, (3, 3))
        flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
        data = paddle.rand([1, 3, 16, 14])
        with paddle.amp.auto_cast(level='O2'):
            conv_out = conv(data)
            out = flatten(conv_out)
        self.assertEqual(conv_out.shape, [1, 8, 14, 12])
        self.assertEqual(out.shape, [1, 112, 12])

    def test_argmax_op_transposer_keep_dims(self):
        if False:
            for i in range(10):
                print('nop')
        conv = paddle.nn.Conv2D(3, 8, (3, 3))
        data = paddle.rand([1, 3, 16, 14])
        with paddle.amp.auto_cast(level='O2'):
            conv_out = conv(data)
            out = paddle.argmax(conv_out, axis=1, keepdim=True)
        self.assertEqual(conv_out.shape, [1, 8, 14, 12])
        self.assertEqual(out.shape, [1, 1, 14, 12])

    def test_concat_op_transposer(self):
        if False:
            return 10
        in1 = paddle.rand([1, 8, 14, 12])
        conv = paddle.nn.Conv2D(3, 8, (3, 3))
        data = paddle.rand([1, 3, 16, 14])
        with paddle.amp.auto_cast(level='O2'):
            conv_out = conv(data)
            out = paddle.concat(x=[conv_out, in1], axis=0)
        self.assertEqual(conv_out.shape, [1, 8, 14, 12])
        self.assertEqual(out.shape, [2, 8, 14, 12])

    def test_concat_op_no_transposer(self):
        if False:
            while True:
                i = 10
        conv = paddle.nn.Conv2D(3, 8, (3, 3))
        data1 = paddle.rand([1, 3, 16, 14])
        data2 = paddle.rand([1, 3, 16, 14])
        with paddle.amp.auto_cast(level='O2'):
            conv_out1 = conv(data1)
            conv_out2 = conv(data2)
            out = paddle.concat(x=[conv_out1, conv_out2], axis=0)
        self.assertEqual(conv_out1.shape, [1, 8, 14, 12])
        self.assertEqual(out.shape, [2, 8, 14, 12])

    def test_padding_tranpose(self):
        if False:
            print('Hello World!')
        conv = paddle.nn.Conv2D(3, 8, (3, 3))
        data = paddle.rand([1, 3, 16, 14])
        mode = 'constant'
        pad = [1, 0, 1, 2]
        padding = paddle.nn.Pad2D(padding=pad, mode=mode, data_format='NCHW')
        with paddle.amp.auto_cast(level='O2', dtype='bfloat16'):
            conv_out = conv(data)
            out = padding(conv_out)
        self.assertEqual(conv_out.shape, [1, 8, 14, 12])
        self.assertEqual(out.shape, [1, 8, 17, 13])

class TestAutoTuneAPI(unittest.TestCase):

    def test_set_config_warnings(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings(record=True) as w:
            config = {'layout': {'enable': 1}}
            tfile = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            json.dump(config, tfile)
            tfile.close()
            paddle.incubate.autotune.set_config(tfile.name)
            os.remove(tfile.name)
            self.assertTrue(len(w) == 1)
if __name__ == '__main__':
    unittest.main()
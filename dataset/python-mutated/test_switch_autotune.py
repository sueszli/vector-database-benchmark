import json
import os
import tempfile
import unittest
import warnings
import numpy as np
import paddle

class SimpleNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.conv = paddle.nn.Conv2D(1, 2, (3, 3))

    def forward(self, image, label=None):
        if False:
            i = 10
            return i + 15
        return self.conv(image)

def train_dygraph(net, data):
    if False:
        while True:
            i = 10
    data.stop_gradient = False
    out = net(data)
    loss = paddle.mean(out)
    adam = paddle.optimizer.Adam(parameters=net.parameters())
    out.backward()
    adam.step()
    adam.clear_grad()

def static_program(net, data):
    if False:
        i = 10
        return i + 15
    data.stop_gradient = False
    out = net(data)
    loss = paddle.mean(out)
    adam = paddle.optimizer.Adam()
    adam.minimize(loss)
    return loss

class TestAutoTune(unittest.TestCase):

    def set_flags(self, enable_autotune):
        if False:
            print('Hello World!')
        if paddle.is_compiled_with_cuda():
            if enable_autotune:
                paddle.set_flags({'FLAGS_conv_workspace_size_limit': -1})
            else:
                paddle.set_flags({'FLAGS_conv_workspace_size_limit': 512})

    def get_flags(self, name):
        if False:
            while True:
                i = 10
        res = paddle.get_flags(name)
        return res[name]

    def get_expected_res(self, step_id, enable_autotune):
        if False:
            for i in range(10):
                print('nop')
        expected_res = {'step_id': step_id, 'cache_size': 0, 'cache_hit_rate': 0}
        if paddle.is_compiled_with_cuda():
            expected_res['cache_size'] = 3
            expected_res['cache_hit_rate'] = (step_id + 0.0) / (step_id + 1.0)
        return expected_res

    def test_autotune(self):
        if False:
            while True:
                i = 10
        paddle.incubate.autotune.set_config(config={'kernel': {'enable': False}})
        self.assertEqual(self.get_flags('FLAGS_use_autotune'), False)
        paddle.incubate.autotune.set_config(config={'kernel': {'enable': True}})
        self.assertEqual(self.get_flags('FLAGS_use_autotune'), True)

    def check_status(self, expected_res):
        if False:
            return 10
        status = paddle.base.core.autotune_status()
        for key in status.keys():
            v = status[key]
            if key == 'cache_hit_rate':
                np.testing.assert_allclose(v, expected_res[key])
            else:
                np.testing.assert_array_equal(v, expected_res[key])

class TestDygraphAutoTuneStatus(TestAutoTune):

    def run_program(self, enable_autotune):
        if False:
            for i in range(10):
                print('nop')
        self.set_flags(enable_autotune)
        if enable_autotune:
            paddle.incubate.autotune.set_config(config={'kernel': {'enable': True, 'tuning_range': [1, 2]}})
        else:
            paddle.incubate.autotune.set_config(config={'kernel': {'enable': False}})
        x_var = paddle.uniform((1, 1, 8, 8), dtype='float32', min=-1.0, max=1.0)
        net = SimpleNet()
        for i in range(3):
            train_dygraph(net, x_var)
            expected_res = self.get_expected_res(i, enable_autotune)
            self.check_status(expected_res)

    def test_enable_autotune(self):
        if False:
            i = 10
            return i + 15
        self.run_program(enable_autotune=True)

    def test_disable_autotune(self):
        if False:
            print('Hello World!')
        self.run_program(enable_autotune=False)

class TestStaticAutoTuneStatus(TestAutoTune):

    def run_program(self, enable_autotune):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        data_shape = [1, 1, 8, 8]
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data = paddle.static.data(name='X', shape=data_shape, dtype='float32')
            net = SimpleNet()
            loss = static_program(net, data)
        place = paddle.CUDAPlace(0) if paddle.base.core.is_compiled_with_cuda() else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(startup_program)
        x = np.random.random(size=data_shape).astype('float32')
        exe.run(program=main_program, feed={'X': x}, fetch_list=[loss])
        self.set_flags(enable_autotune)
        if enable_autotune:
            config = {'kernel': {'enable': True, 'tuning_range': [1, 2]}}
            tfile = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            json.dump(config, tfile)
            tfile.close()
            paddle.incubate.autotune.set_config(tfile.name)
            os.remove(tfile.name)
        else:
            paddle.incubate.autotune.set_config(config={'kernel': {'enable': False, 'tuning_range': [1, 2]}})
        for i in range(3):
            exe.run(program=main_program, feed={'X': x}, fetch_list=[loss])
            status = paddle.base.core.autotune_status()
            expected_res = self.get_expected_res(i, enable_autotune)
            self.check_status(expected_res)
        paddle.disable_static()

    def func_enable_autotune(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_program(enable_autotune=True)

    def test_enable_autotune(self):
        if False:
            for i in range(10):
                print('nop')
        self.func_enable_autotune()

    def func_disable_autotune(self):
        if False:
            while True:
                i = 10
        self.run_program(enable_autotune=False)

    def test_disable_autotune(self):
        if False:
            while True:
                i = 10
        self.func_disable_autotune()

class TestAutoTuneAPI(unittest.TestCase):

    def test_set_config_warnings(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings(record=True) as w:
            config = {'kernel': {'enable': 1, 'tuning_range': 1}}
            tfile = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            json.dump(config, tfile)
            tfile.close()
            paddle.incubate.autotune.set_config(tfile.name)
            os.remove(tfile.name)
            self.assertTrue(len(w) == 2)

    def test_set_config_attr(self):
        if False:
            while True:
                i = 10
        paddle.incubate.autotune.set_config(config=None)
        self.assertEqual(paddle.get_flags('FLAGS_use_autotune')['FLAGS_use_autotune'], True)
if __name__ == '__main__':
    unittest.main()
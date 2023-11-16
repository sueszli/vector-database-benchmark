import os
import tempfile
import unittest
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn, profiler
from paddle.io import DataLoader, Dataset
from paddle.profiler import utils

class TestProfiler(unittest.TestCase):

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir.cleanup()

    def test_profiler(self):
        if False:
            while True:
                i = 10

        def my_trace_back(prof):
            if False:
                return 10
            path = os.path.join(self.temp_dir.name, './test_profiler_chrometracing')
            profiler.export_chrome_tracing(path)(prof)
            path = os.path.join(self.temp_dir.name, './test_profiler_pb')
            profiler.export_protobuf(path)(prof)
        self.temp_dir = tempfile.TemporaryDirectory()
        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(x_value, stop_gradient=False, place=paddle.CPUPlace())
        y = x / 2.0
        ones_like_y = paddle.ones_like(y)
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU]) as prof:
            y = x / 2.0
        prof = None
        self.assertEqual(utils._is_profiler_used, False)
        with profiler.RecordEvent(name='test'):
            y = x / 2.0
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU], scheduler=(1, 2)) as prof:
            self.assertEqual(utils._is_profiler_used, True)
            with profiler.RecordEvent(name='test'):
                y = x / 2.0
        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU], scheduler=profiler.make_scheduler(closed=0, ready=1, record=1, repeat=1), on_trace_ready=my_trace_back) as prof:
            y = x / 2.0
        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU], scheduler=profiler.make_scheduler(closed=0, ready=0, record=2, repeat=1), on_trace_ready=my_trace_back) as prof:
            for i in range(3):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU], scheduler=lambda x: profiler.ProfilerState.RECORD_AND_RETURN, on_trace_ready=my_trace_back, with_flops=True) as prof:
            for i in range(2):
                y = x / 2.0
                prof.step()

        def my_scheduler(num_step):
            if False:
                print('Hello World!')
            if num_step % 5 < 2:
                return profiler.ProfilerState.RECORD_AND_RETURN
            elif num_step % 5 < 3:
                return profiler.ProfilerState.READY
            elif num_step % 5 < 4:
                return profiler.ProfilerState.RECORD
            else:
                return profiler.ProfilerState.CLOSED

        def my_scheduler1(num_step):
            if False:
                while True:
                    i = 10
            if num_step % 5 < 2:
                return profiler.ProfilerState.RECORD
            elif num_step % 5 < 3:
                return profiler.ProfilerState.READY
            elif num_step % 5 < 4:
                return profiler.ProfilerState.RECORD
            else:
                return profiler.ProfilerState.CLOSED
        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU], scheduler=lambda x: profiler.ProfilerState.RECORD_AND_RETURN, on_trace_ready=my_trace_back) as prof:
            for i in range(2):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU], scheduler=my_scheduler, on_trace_ready=my_trace_back) as prof:
            for i in range(5):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU], scheduler=my_scheduler1) as prof:
            for i in range(5):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU], scheduler=profiler.make_scheduler(closed=1, ready=1, record=2, repeat=1, skip_first=1), on_trace_ready=my_trace_back, profile_memory=True, record_shapes=True) as prof:
            for i in range(5):
                y = x / 2.0
                paddle.grad(outputs=y, inputs=[x], grad_outputs=ones_like_y)
                prof.step()
        path = os.path.join(self.temp_dir.name, './test_profiler_pb.pb')
        prof.export(path=path, format='pb')
        prof.summary()
        result = profiler.utils.load_profiler_result(path)
        prof = None
        dataset = RandomDataset(10 * 4)
        simple_net = SimpleNet()
        opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=simple_net.parameters())
        loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=2)
        prof = profiler.Profiler(on_trace_ready=lambda prof: None)
        prof.start()
        for (i, (image, label)) in enumerate(loader()):
            out = simple_net(image)
            loss = F.cross_entropy(out, label)
            avg_loss = paddle.mean(loss)
            avg_loss.backward()
            opt.minimize(avg_loss)
            simple_net.clear_gradients()
            prof.step()
        prof.stop()
        prof.summary()
        prof = None
        dataset = RandomDataset(10 * 4)
        simple_net = SimpleNet()
        loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
        opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=simple_net.parameters())
        prof = profiler.Profiler(on_trace_ready=lambda prof: None)
        prof.start()
        for (i, (image, label)) in enumerate(loader()):
            out = simple_net(image)
            loss = F.cross_entropy(out, label)
            avg_loss = paddle.mean(loss)
            avg_loss.backward()
            opt.step()
            simple_net.clear_gradients()
            prof.step()
        prof.stop()

class TestGetProfiler(unittest.TestCase):

    def test_getprofiler(self):
        if False:
            print('Hello World!')
        config_content = '\n        {\n        "targets": ["CPU"],\n        "scheduler": [3,4],\n        "on_trace_ready": {\n            "export_chrome_tracing":{\n                "module": "paddle.profiler",\n                "use_direct": false,\n                "args": [],\n                "kwargs": {\n                        "dir_name": "testdebug/"\n                    }\n                }\n            },\n          "timer_only": false\n        }\n        '
        filehandle = tempfile.NamedTemporaryFile(mode='w')
        filehandle.write(config_content)
        filehandle.flush()
        from paddle.profiler import profiler
        profiler = profiler.get_profiler(filehandle.name)
        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(x_value, stop_gradient=False, place=paddle.CPUPlace())
        with profiler:
            for i in range(5):
                y = x / 2.0
                ones_like_y = paddle.ones_like(y)
                profiler.step()
        config_content = '\n        {\n        "targets": ["Cpu", "Gpu"],\n        "scheduler": {\n            "make_scheduler":{\n                "module": "paddle.profiler",\n                "use_direct": true,\n                "args": [],\n                "kwargs": {}\n            }\n        },\n        "on_trace_ready": {\n            "export_chrome_tracing":{\n                "module": "paddle.profiler1",\n                "use_direct": true,\n                "args": [],\n                "kwargs": {\n                    }\n                }\n            },\n          "timer_only": false\n        }\n        '
        filehandle = tempfile.NamedTemporaryFile(mode='w')
        filehandle.write(config_content)
        filehandle.flush()
        from paddle.profiler import profiler
        try:
            profiler = profiler.get_profiler(filehandle.name)
        except:
            pass
        config_content = '\n        {\n        "targets": ["Cpu", "Gpu"],\n        "scheduler": {\n           "make_scheduler":{\n                "module": "paddle.profiler",\n                "use_direct": false,\n                "args": [],\n                "kwargs": {\n                        "closed": 1,\n                        "ready": 1,\n                        "record": 2\n                    }\n            }\n        },\n        "on_trace_ready": {\n            "export_chrome_tracing":{\n                "module": "paddle.profiler",\n                "use_direct": true,\n                "args": [],\n                "kwargs": {\n                    }\n                }\n            },\n          "timer_only": false\n        }\n        '
        filehandle = tempfile.NamedTemporaryFile(mode='w')
        filehandle.write(config_content)
        filehandle.flush()
        from paddle.profiler import profiler
        profiler = profiler.get_profiler(filehandle.name)
        config_content = '\n        {\n        "targets": [1],\n        "scheduler": {\n            "make_scheduler1":{\n                "module": "paddle.profiler",\n                "use_direct": false,\n                "args": [],\n                "kwargs": {\n                        "closed": 1,\n                        "ready": 1,\n                        "record": 2\n                    }\n            }\n        },\n        "on_trace_ready": {\n            "export_chrome_tracing1":{\n                "module": "paddle.profiler",\n                "use_direct": false,\n                "args": [],\n                "kwargs": {\n                        "dir_name": "testdebug/"\n                    }\n                }\n            },\n          "timer_only": 1\n        }\n        '
        filehandle = tempfile.NamedTemporaryFile(mode='w')
        filehandle.write(config_content)
        filehandle.flush()
        from paddle.profiler import profiler
        profiler = profiler.get_profiler(filehandle.name)
        from paddle.profiler import profiler
        profiler = profiler.get_profiler('nopath.json')

class RandomDataset(Dataset):

    def __init__(self, num_samples):
        if False:
            while True:
                i = 10
        self.num_samples = num_samples

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        image = np.random.random([100]).astype('float32')
        label = np.random.randint(0, 10 - 1, (1,)).astype('int64')
        return (image, label)

    def __len__(self):
        if False:
            print('Hello World!')
        return self.num_samples

class SimpleNet(nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, image, label=None):
        if False:
            i = 10
            return i + 15
        return self.fc(image)

class TestTimerOnly(unittest.TestCase):

    def test_with_dataloader(self):
        if False:
            for i in range(10):
                print('nop')

        def train(step_num_samples=None):
            if False:
                for i in range(10):
                    print('nop')
            dataset = RandomDataset(20 * 4)
            simple_net = SimpleNet()
            opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=simple_net.parameters())
            loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=2)
            step_info = ''
            p = profiler.Profiler(timer_only=True)
            p.start()
            for (i, (image, label)) in enumerate(loader()):
                out = simple_net(image)
                loss = F.cross_entropy(out, label)
                avg_loss = paddle.mean(loss)
                avg_loss.backward()
                opt.minimize(avg_loss)
                simple_net.clear_gradients()
                p.step(num_samples=step_num_samples)
                if i % 10 == 0:
                    step_info = p.step_info()
                    print(f'Iter {i}: {step_info}')
            p.stop()
            return step_info
        step_info = train(step_num_samples=None)
        self.assertTrue('steps/s' in step_info)
        step_info = train(step_num_samples=4)
        self.assertTrue('samples/s' in step_info)

    def test_without_dataloader(self):
        if False:
            while True:
                i = 10
        x = paddle.to_tensor(np.random.randn(10, 10))
        y = paddle.to_tensor(np.random.randn(10, 10))
        p = profiler.Profiler(timer_only=True)
        p.start()
        step_info = ''
        for i in range(20):
            out = x + y
            p.step()
        p.stop()
if __name__ == '__main__':
    unittest.main()
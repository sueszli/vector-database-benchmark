import os
import sys
import tempfile
import unittest

class TestCustomCPUProfilerPlugin(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory()
        cmd = 'cd {}             && git clone --depth 1 {}             && cd PaddleCustomDevice             && git fetch origin             && git checkout {} -b dev             && cd backends/custom_cpu             && mkdir build && cd build && cmake .. -DPython_EXECUTABLE={} -DWITH_TESTING=OFF && make -j8'.format(self.temp_dir.name, os.getenv('PLUGIN_URL'), os.getenv('PLUGIN_TAG'), sys.executable)
        os.system(cmd)
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.join(cur_dir, f'{self.temp_dir.name}/PaddleCustomDevice/backends/custom_cpu/build')

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()
        del os.environ['CUSTOM_DEVICE_ROOT']

    def test_custom_profiler(self):
        if False:
            for i in range(10):
                print('nop')
        import paddle
        from paddle import profiler
        paddle.set_device('custom_cpu')
        x = paddle.to_tensor([1, 2, 3])
        p = profiler.Profiler(targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.CUSTOM_DEVICE])
        p.start()
        for iter in range(10):
            x = x + 1
            p.step()
        p.stop()
        p.summary()
if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        sys.exit()
    unittest.main()
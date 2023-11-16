"""Tests for tf 2.x profiler."""
import os
import socket
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.profiler import trace

class ProfilerTest(test_util.TensorFlowTestCase):

    def test_profile_exceptions(self):
        if False:
            print('Hello World!')
        logdir = self.get_temp_dir()
        profiler.start(logdir)
        with self.assertRaises(errors.AlreadyExistsError):
            profiler.start(logdir)
        profiler.stop()
        with self.assertRaises(errors.UnavailableError):
            profiler.stop()
        profiler.start('/dev/null/\\/\\/:123')
        with self.assertRaises(Exception):
            profiler.stop()
        profiler.start(logdir)
        profiler.stop()

    def test_save_profile(self):
        if False:
            while True:
                i = 10
        logdir = self.get_temp_dir()
        profiler.start(logdir)
        with trace.Trace('three_times_five'):
            three = constant_op.constant(3)
            five = constant_op.constant(5)
            product = three * five
        self.assertAllEqual(15, product)
        profiler.stop()
        file_list = gfile.ListDirectory(logdir)
        self.assertEqual(len(file_list), 1)
        for file_name in gfile.ListDirectory(logdir):
            if gfile.IsDirectory(os.path.join(logdir, file_name)):
                self.assertEqual(file_name, 'plugins')
        profile_dir = os.path.join(logdir, 'plugins', 'profile')
        run = gfile.ListDirectory(profile_dir)[0]
        hostname = socket.gethostname()
        xplane = os.path.join(profile_dir, run, hostname + '.xplane.pb')
        self.assertTrue(gfile.Exists(xplane))

    def test_profile_with_options(self):
        if False:
            for i in range(10):
                print('nop')
        logdir = self.get_temp_dir()
        options = profiler.ProfilerOptions(host_tracer_level=3, python_tracer_level=1)
        profiler.start(logdir, options)
        with trace.Trace('three_times_five'):
            three = constant_op.constant(3)
            five = constant_op.constant(5)
            product = three * five
        self.assertAllEqual(15, product)
        profiler.stop()
        file_list = gfile.ListDirectory(logdir)
        self.assertEqual(len(file_list), 1)

    def test_context_manager_with_options(self):
        if False:
            return 10
        logdir = self.get_temp_dir()
        options = profiler.ProfilerOptions(host_tracer_level=3, python_tracer_level=1)
        with profiler.Profile(logdir, options):
            with trace.Trace('three_times_five'):
                three = constant_op.constant(3)
                five = constant_op.constant(5)
                product = three * five
            self.assertAllEqual(15, product)
        file_list = gfile.ListDirectory(logdir)
        self.assertEqual(len(file_list), 1)
if __name__ == '__main__':
    test.main()
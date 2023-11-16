"""Tests for tf 2.x profiler."""
import glob
import os
import threading
import portpicker
from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_client
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.profiler.integration_test import mnist_testing_utils

def _model_setup():
    if False:
        for i in range(10):
            print('nop')
    'Set up a MNIST Keras model for testing purposes.\n\n  Builds a MNIST Keras model and returns model information.\n\n  Returns:\n    A tuple of (batch_size, steps, train_dataset, mode)\n  '
    context.set_log_device_placement(True)
    batch_size = 64
    steps = 2
    with collective_strategy.CollectiveAllReduceStrategy().scope():
        (train_ds, _) = mnist_testing_utils.mnist_synthetic_dataset(batch_size, steps)
        model = mnist_testing_utils.get_mnist_model((28, 28, 1))
    return (batch_size, steps, train_ds, model)

def _make_temp_log_dir(test_obj):
    if False:
        i = 10
        return i + 15
    return test_obj.get_temp_dir()

class ProfilerApiTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.worker_start = threading.Event()
        self.profile_done = False

    def _check_xspace_pb_exist(self, logdir):
        if False:
            print('Hello World!')
        path = os.path.join(logdir, 'plugins', 'profile', '*', '*.xplane.pb')
        self.assertEqual(1, len(glob.glob(path)), 'Expected one path match: ' + path)

    def test_single_worker_no_profiling(self):
        if False:
            print('Hello World!')
        'Test single worker without profiling.'
        (_, steps, train_ds, model) = _model_setup()
        model.fit(x=train_ds, epochs=2, steps_per_epoch=steps)

    def test_single_worker_sampling_mode(self, delay_ms=None):
        if False:
            while True:
                i = 10
        'Test single worker sampling mode.'

        def on_worker(port, worker_start):
            if False:
                print('Hello World!')
            logging.info('worker starting server on {}'.format(port))
            profiler.start_server(port)
            (_, steps, train_ds, model) = _model_setup()
            worker_start.set()
            while True:
                model.fit(x=train_ds, epochs=2, steps_per_epoch=steps)
                if self.profile_done:
                    break

        def on_profile(port, logdir, worker_start):
            if False:
                i = 10
                return i + 15
            duration_ms = 30
            worker_start.wait()
            options = profiler.ProfilerOptions(host_tracer_level=2, python_tracer_level=0, device_tracer_level=1, delay_ms=delay_ms)
            profiler_client.trace('localhost:{}'.format(port), logdir, duration_ms, '', 100, options)
            self.profile_done = True
        logdir = self.get_temp_dir()
        port = portpicker.pick_unused_port()
        thread_profiler = threading.Thread(target=on_profile, args=(port, logdir, self.worker_start))
        thread_worker = threading.Thread(target=on_worker, args=(port, self.worker_start))
        thread_worker.start()
        thread_profiler.start()
        thread_profiler.join()
        thread_worker.join(120)
        self._check_xspace_pb_exist(logdir)

    def test_single_worker_sampling_mode_short_delay(self):
        if False:
            print('Hello World!')
        'Test single worker sampling mode with a short delay.\n\n    Expect that requested delayed start time will arrive late, and a subsequent\n    retry will issue an immediate start.\n    '
        self.test_single_worker_sampling_mode(delay_ms=1)

    def test_single_worker_sampling_mode_long_delay(self):
        if False:
            for i in range(10):
                print('nop')
        'Test single worker sampling mode with a long delay.'
        self.test_single_worker_sampling_mode(delay_ms=1000)

    def test_single_worker_programmatic_mode(self):
        if False:
            while True:
                i = 10
        'Test single worker programmatic mode.'
        logdir = self.get_temp_dir()
        options = profiler.ProfilerOptions(host_tracer_level=2, python_tracer_level=0, device_tracer_level=1)
        profiler.start(logdir, options)
        (_, steps, train_ds, model) = _model_setup()
        model.fit(x=train_ds, epochs=2, steps_per_epoch=steps)
        profiler.stop()
        self._check_xspace_pb_exist(logdir)
if __name__ == '__main__':
    multi_process_runner.test_main()
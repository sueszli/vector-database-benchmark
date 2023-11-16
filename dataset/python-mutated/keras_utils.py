"""Helper functions for the Keras implementations of models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import profiler

class BatchTimestamp(object):
    """A structure to store batch time stamp."""

    def __init__(self, batch_index, timestamp):
        if False:
            return 10
        self.batch_index = batch_index
        self.timestamp = timestamp

    def __repr__(self):
        if False:
            return 10
        return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(self.batch_index, self.timestamp)

class TimeHistory(tf.keras.callbacks.Callback):
    """Callback for Keras models."""

    def __init__(self, batch_size, log_steps):
        if False:
            print('Hello World!')
        'Callback for logging performance.\n\n    Args:\n      batch_size: Total batch size.\n      log_steps: Interval of steps between logging of batch level stats.\n    '
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.log_steps = log_steps
        self.global_steps = 0
        self.timestamp_log = []
        self.epoch_runtime_log = []

    def on_train_end(self, logs=None):
        if False:
            for i in range(10):
                print('nop')
        self.train_finish_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        if False:
            return 10
        self.epoch_start = time.time()

    def on_batch_begin(self, batch, logs=None):
        if False:
            return 10
        self.global_steps += 1
        if self.global_steps == 1:
            self.start_time = time.time()
            self.timestamp_log.append(BatchTimestamp(self.global_steps, self.start_time))

    def on_batch_end(self, batch, logs=None):
        if False:
            print('Hello World!')
        'Records elapse time of the batch and calculates examples per second.'
        if self.global_steps % self.log_steps == 0:
            timestamp = time.time()
            elapsed_time = timestamp - self.start_time
            examples_per_second = self.batch_size * self.log_steps / elapsed_time
            self.timestamp_log.append(BatchTimestamp(self.global_steps, timestamp))
            tf.compat.v1.logging.info("BenchmarkMetric: {'global step':%d, 'time_taken': %f,'examples_per_second': %f}" % (self.global_steps, elapsed_time, examples_per_second))
            self.start_time = timestamp

    def on_epoch_end(self, epoch, logs=None):
        if False:
            return 10
        epoch_run_time = time.time() - self.epoch_start
        self.epoch_runtime_log.append(epoch_run_time)
        tf.compat.v1.logging.info("BenchmarkMetric: {'epoch':%d, 'time_taken': %f}" % (epoch, epoch_run_time))

def get_profiler_callback(model_dir, profile_steps, enable_tensorboard):
    if False:
        print('Hello World!')
    'Validate profile_steps flag value and return profiler callback.'
    profile_steps_error_message = 'profile_steps must be a comma separated pair of positive integers, specifying the first and last steps to be profiled.'
    try:
        profile_steps = [int(i) for i in profile_steps.split(',')]
    except ValueError:
        raise ValueError(profile_steps_error_message)
    if len(profile_steps) != 2:
        raise ValueError(profile_steps_error_message)
    (start_step, stop_step) = profile_steps
    if start_step < 0 or start_step > stop_step:
        raise ValueError(profile_steps_error_message)
    if enable_tensorboard:
        tf.compat.v1.logging.warn('Both TensorBoard and profiler callbacks are used. Note that the TensorBoard callback profiles the 2nd step (unless otherwise specified). Please make sure the steps profiled by the two callbacks do not overlap.')
    return ProfilerCallback(model_dir, start_step, stop_step)

class ProfilerCallback(tf.keras.callbacks.Callback):
    """Save profiles in specified step range to log directory."""

    def __init__(self, log_dir, start_step, stop_step):
        if False:
            i = 10
            return i + 15
        super(ProfilerCallback, self).__init__()
        self.log_dir = log_dir
        self.start_step = start_step
        self.stop_step = stop_step

    def on_batch_begin(self, batch, logs=None):
        if False:
            for i in range(10):
                print('nop')
        if batch == self.start_step:
            profiler.start()
            tf.compat.v1.logging.info('Profiler started at Step %s', self.start_step)

    def on_batch_end(self, batch, logs=None):
        if False:
            print('Hello World!')
        if batch == self.stop_step:
            results = profiler.stop()
            profiler.save(self.log_dir, results)
            tf.compat.v1.logging.info('Profiler saved profiles for steps between %s and %s to %s', self.start_step, self.stop_step, self.log_dir)

def set_session_config(enable_eager=False, enable_xla=False):
    if False:
        while True:
            i = 10
    'Sets the session config.'
    if is_v2_0():
        set_config_v2(enable_xla=enable_xla)
    else:
        config = get_config_proto_v1(enable_xla=enable_xla)
        if enable_eager:
            tf.compat.v1.enable_eager_execution(config=config)
        else:
            sess = tf.Session(config=config)
            tf.keras.backend.set_session(sess)

def get_config_proto_v1(enable_xla=False):
    if False:
        while True:
            i = 10
    'Return config proto according to flag settings, or None to use default.'
    config = None
    if enable_xla:
        config = tf.compat.v1.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
        config.graph_options.rewrite_options.pin_to_host_optimization = rewriter_config_pb2.RewriterConfig.OFF
    return config

def set_config_v2(enable_xla=False):
    if False:
        for i in range(10):
            print('nop')
    'Config eager context according to flag values using TF 2.0 API.'
    if enable_xla:
        tf.config.optimizer.set_jit(True)
        tf.config.optimizer.set_experimental_options({'pin_to_host_optimization': False})

def is_v2_0():
    if False:
        i = 10
        return i + 15
    'Returns true if using tf 2.0.'
    return tf2.enabled()
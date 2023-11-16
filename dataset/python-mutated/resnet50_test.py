"""Tests and benchmarks for the ResNet50 model, executed eagerly."""
import gc
import os
import tempfile
import time
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager.benchmarks.resnet50 import resnet50
from tensorflow.python.eager.benchmarks.resnet50 import resnet50_test_util
from tensorflow.python.framework import test_util

def compute_gradients(model, images, labels, num_replicas=1):
    if False:
        for i in range(10):
            print('nop')
    with tf.GradientTape() as grad_tape:
        logits = model(images, training=True)
        loss = tf.compat.v1.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
        tf.compat.v2.summary.write('loss', loss)
        if num_replicas != 1:
            loss /= num_replicas
    with record.stop_recording():
        grads = grad_tape.gradient(loss, model.variables)
    return grads

def apply_gradients(model, optimizer, gradients):
    if False:
        for i in range(10):
            print('nop')
    optimizer.apply_gradients(zip(gradients, model.variables))

def _events_from_file(filepath):
    if False:
        print('Hello World!')
    'Returns all events in a single event file.\n\n  Args:\n    filepath: Path to the event file.\n\n  Returns:\n    A list of all tf.compat.v1.Event protos in the event file.\n  '
    records = list(tf.compat.v1.python_io.tf_record_iterator(filepath))
    result = []
    for r in records:
        event = tf.compat.v1.Event()
        event.ParseFromString(r)
        result.append(event)
    return result

def events_from_logdir(logdir):
    if False:
        for i in range(10):
            print('nop')
    'Returns all events in the single eventfile in logdir.\n\n  Args:\n    logdir: The directory in which the single event file is sought.\n\n  Returns:\n    A list of all tf.compat.v1.Event protos from the single event file.\n\n  Raises:\n    AssertionError: If logdir does not contain exactly one file.\n  '
    assert tf.io.gfile.exists(logdir)
    files = tf.io.gfile.listdir(logdir)
    assert len(files) == 1, 'Found not exactly one file in logdir: %s' % files
    return _events_from_file(os.path.join(logdir, files[0]))

class ResNet50Test(tf.test.TestCase):

    def _apply(self, defun=False, execution_mode=None):
        if False:
            for i in range(10):
                print('nop')
        (device, data_format) = resnet50_test_util.device_and_data_format()
        model = resnet50.ResNet50(data_format)
        if defun:
            model.call = tf.function(model.call)
        with tf.device(device), context.execution_mode(execution_mode):
            (images, _) = resnet50_test_util.random_batch(2, data_format)
            output = model(images, training=False)
            context.async_wait()
        self.assertEqual((2, 1000), output.shape)

    @test_util.disable_tfrt('Flaky test. b/157103729')
    def test_apply(self):
        if False:
            for i in range(10):
                print('nop')
        self._apply(defun=False)

    def test_apply_async(self):
        if False:
            for i in range(10):
                print('nop')
        self._apply(defun=False, execution_mode=context.ASYNC)

    def test_apply_with_defun(self):
        if False:
            return 10
        self._apply(defun=True)

    def test_apply_with_defun_async(self):
        if False:
            return 10
        self._apply(defun=True, execution_mode=context.ASYNC)

    @test_util.disable_tfrt('Flaky test. b/157103729')
    def test_apply_no_top(self):
        if False:
            print('Hello World!')
        (device, data_format) = resnet50_test_util.device_and_data_format()
        model = resnet50.ResNet50(data_format, include_top=False)
        with tf.device(device):
            (images, _) = resnet50_test_util.random_batch(2, data_format)
            output = model(images, training=False)
        output_shape = (2, 2048, 1, 1) if data_format == 'channels_first' else (2, 1, 1, 2048)
        self.assertEqual(output_shape, output.shape)

    @test_util.disable_tfrt('Flaky test. b/157103729')
    def test_apply_with_pooling(self):
        if False:
            while True:
                i = 10
        (device, data_format) = resnet50_test_util.device_and_data_format()
        model = resnet50.ResNet50(data_format, include_top=False, pooling='avg')
        with tf.device(device):
            (images, _) = resnet50_test_util.random_batch(2, data_format)
            output = model(images, training=False)
        self.assertEqual((2, 2048), output.shape)

    @test_util.disable_tfrt('Flaky test. b/157103729')
    def test_apply_no_average_pooling(self):
        if False:
            print('Hello World!')
        (device, data_format) = resnet50_test_util.device_and_data_format()
        model = resnet50.ResNet50(data_format, average_pooling=False, include_top=False)
        with tf.device(device):
            (images, _) = resnet50_test_util.random_batch(2, data_format)
            output = model(images, training=False)
        output_shape = (2, 2048, 7, 7) if data_format == 'channels_first' else (2, 7, 7, 2048)
        self.assertEqual(output_shape, output.shape)

    @test_util.disable_tfrt('Flaky test. b/157103729')
    def test_apply_block3_strides(self):
        if False:
            return 10
        (device, data_format) = resnet50_test_util.device_and_data_format()
        model = resnet50.ResNet50(data_format, block3_strides=True, include_top=False)
        with tf.device(device):
            (images, _) = resnet50_test_util.random_batch(2, data_format)
            output = model(images, training=False)
        output_shape = (2, 2048, 1, 1) if data_format == 'channels_first' else (2, 1, 1, 2048)
        self.assertEqual(output_shape, output.shape)

    @test_util.disable_tfrt('Flaky test. b/157103729')
    def test_apply_retrieve_intermediates(self):
        if False:
            for i in range(10):
                print('nop')
        (device, data_format) = resnet50_test_util.device_and_data_format()
        model = resnet50.ResNet50(data_format, block3_strides=True, include_top=False)
        intermediates_dict = {}
        with tf.device(device):
            (images, _) = resnet50_test_util.random_batch(2, data_format)
            output = model(images, training=False, intermediates_dict=intermediates_dict)
        output_shape = (2, 2048, 1, 1) if data_format == 'channels_first' else (2, 1, 1, 2048)
        self.assertEqual(output_shape, output.shape)
        if data_format == 'channels_first':
            block_shapes = {'block0': (2, 64, 112, 112), 'block0mp': (2, 64, 55, 55), 'block1': (2, 256, 55, 55), 'block2': (2, 512, 28, 28), 'block3': (2, 1024, 7, 7), 'block4': (2, 2048, 1, 1)}
        else:
            block_shapes = {'block0': (2, 112, 112, 64), 'block0mp': (2, 55, 55, 64), 'block1': (2, 55, 55, 256), 'block2': (2, 28, 28, 512), 'block3': (2, 7, 7, 1024), 'block4': (2, 1, 1, 2048)}
        for (block_name, block) in intermediates_dict.items():
            self.assertEqual(block_shapes[block_name], block.shape)

    def _test_train(self, execution_mode=None):
        if False:
            while True:
                i = 10
        (device, data_format) = resnet50_test_util.device_and_data_format()
        model = resnet50.ResNet50(data_format)
        tf.compat.v2.summary.experimental.set_step(tf.compat.v1.train.get_or_create_global_step())
        logdir = tempfile.mkdtemp()
        with tf.compat.v2.summary.create_file_writer(logdir, max_queue=0, name='t0').as_default(), tf.compat.v2.summary.record_if(True):
            with tf.device(device), context.execution_mode(execution_mode):
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
                (images, labels) = resnet50_test_util.random_batch(2, data_format)
                apply_gradients(model, optimizer, compute_gradients(model, images, labels))
                self.assertEqual(320, len(model.variables))
                context.async_wait()
        events = events_from_logdir(logdir)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1].summary.value[0].tag, 'loss')

    @test_util.disable_tfrt('Flaky test. b/157103729')
    def test_train(self):
        if False:
            i = 10
            return i + 15
        self._test_train()

    @test_util.disable_tfrt('TFE_ContextGetExecutorForThread missing b/156188669')
    def test_train_async(self):
        if False:
            while True:
                i = 10
        self._test_train(execution_mode=context.ASYNC)

    @test_util.disable_tfrt('Flaky test. b/157103729')
    def test_no_garbage(self):
        if False:
            print('Hello World!')
        (device, data_format) = resnet50_test_util.device_and_data_format()
        model = resnet50.ResNet50(data_format)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
        with tf.device(device):
            (images, labels) = resnet50_test_util.random_batch(2, data_format)
            gc.disable()
            apply_gradients(model, optimizer, compute_gradients(model, images, labels))
            gc.collect()
            previous_gc_debug_flags = gc.get_debug()
            gc.set_debug(gc.DEBUG_SAVEALL)
            for _ in range(2):
                apply_gradients(model, optimizer, compute_gradients(model, images, labels))
            gc.collect()
            self.assertEqual(0, len(gc.garbage))
            gc.set_debug(previous_gc_debug_flags)
            gc.enable()

class MockIterator(object):

    def __init__(self, tensors):
        if False:
            return 10
        self._tensors = [tf.identity(x) for x in tensors]

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        return self._tensors

class ResNet50Benchmarks(tf.test.Benchmark):

    def _report(self, label, start, num_iters, device, batch_size, data_format, num_replicas=1):
        if False:
            print('Hello World!')
        resnet50_test_util.report(self, label, start, num_iters, device, batch_size, data_format, num_replicas)

    def _train_batch_sizes(self):
        if False:
            i = 10
            return i + 15
        'Choose batch sizes based on GPU capability.'
        for device in device_lib.list_local_devices():
            if tf.DeviceSpec.from_string(device.name).device_type == 'GPU':
                if 'K20' in device.physical_device_desc:
                    return (16,)
                if 'P1000' in device.physical_device_desc:
                    return (16,)
                if 'P100' in device.physical_device_desc:
                    return (16, 32, 64)
            if tf.DeviceSpec.from_string(device.name).device_type == 'TPU':
                return (32,)
        return (16, 32)

    def _force_device_sync(self):
        if False:
            for i in range(10):
                print('nop')
        tf.constant(1.0).cpu()

    def _benchmark_eager_apply(self, label, device_and_format, defun=False, execution_mode=None):
        if False:
            while True:
                i = 10
        with context.execution_mode(execution_mode):
            (device, data_format) = device_and_format
            model = resnet50.ResNet50(data_format)
            if defun:
                model.call = tf.function(model.call)
            batch_size = 64
            num_burn = 5
            num_iters = 30
            with tf.device(device):
                (images, _) = resnet50_test_util.random_batch(batch_size, data_format)
                for _ in range(num_burn):
                    model(images, training=False).cpu()
                if execution_mode:
                    context.async_wait()
                gc.collect()
                start = time.time()
                for _ in range(num_iters):
                    model(images, training=False).cpu()
                if execution_mode:
                    context.async_wait()
                self._report(label, start, num_iters, device, batch_size, data_format)

    def benchmark_eager_apply_sync(self):
        if False:
            i = 10
            return i + 15
        self._benchmark_eager_apply('eager_apply', resnet50_test_util.device_and_data_format(), defun=False)

    def benchmark_eager_apply_async(self):
        if False:
            return 10
        self._benchmark_eager_apply('eager_apply_async', resnet50_test_util.device_and_data_format(), defun=False, execution_mode=context.ASYNC)

    def benchmark_eager_apply_with_defun(self):
        if False:
            while True:
                i = 10
        self._benchmark_eager_apply('eager_apply_with_defun', resnet50_test_util.device_and_data_format(), defun=True)

    def _benchmark_eager_train(self, label, make_iterator, device_and_format, defun=False, execution_mode=None):
        if False:
            print('Hello World!')
        with context.execution_mode(execution_mode):
            (device, data_format) = device_and_format
            for batch_size in self._train_batch_sizes():
                (images, labels) = resnet50_test_util.random_batch(batch_size, data_format)
                model = resnet50.ResNet50(data_format)
                optimizer = tf.keras.optimizers.SGD(0.1, 0.1)
                apply_grads = apply_gradients
                if defun:
                    model.call = tf.function(model.call)
                    apply_grads = tf.function(apply_gradients)
                num_burn = 3
                num_iters = 10
                with tf.device(device):
                    iterator = make_iterator((images, labels))
                    for _ in range(num_burn):
                        (images, labels) = iterator.next()
                        apply_grads(model, optimizer, compute_gradients(model, images, labels))
                    if execution_mode:
                        context.async_wait()
                    self._force_device_sync()
                    gc.collect()
                    start = time.time()
                    for _ in range(num_iters):
                        (images, labels) = iterator.next()
                        apply_grads(model, optimizer, compute_gradients(model, images, labels))
                    if execution_mode:
                        context.async_wait()
                    self._force_device_sync()
                    self._report(label, start, num_iters, device, batch_size, data_format)

    def benchmark_eager_train_sync(self):
        if False:
            print('Hello World!')
        self._benchmark_eager_train('eager_train', MockIterator, resnet50_test_util.device_and_data_format(), defun=False)

    def benchmark_eager_train_async(self):
        if False:
            i = 10
            return i + 15
        self._benchmark_eager_train('eager_train_async', MockIterator, resnet50_test_util.device_and_data_format(), defun=False, execution_mode=context.ASYNC)

    def benchmark_eager_train_with_defun(self):
        if False:
            while True:
                i = 10
        self._benchmark_eager_train('eager_train_with_defun', MockIterator, resnet50_test_util.device_and_data_format(), defun=True)

    def benchmark_eager_train_datasets(self):
        if False:
            return 10

        def make_iterator(tensors):
            if False:
                i = 10
                return i + 15
            with tf.device('/device:CPU:0'):
                ds = tf.data.Dataset.from_tensors(tensors).repeat()
            return iter(ds)
        self._benchmark_eager_train('eager_train_dataset', make_iterator, resnet50_test_util.device_and_data_format(), defun=False)

    def benchmark_eager_train_datasets_with_defun(self):
        if False:
            return 10

        def make_iterator(tensors):
            if False:
                while True:
                    i = 10
            with tf.device('/device:CPU:0'):
                ds = tf.data.Dataset.from_tensors(tensors).repeat()
            return iter(ds)
        self._benchmark_eager_train('eager_train_dataset_with_defun', make_iterator, resnet50_test_util.device_and_data_format(), defun=True)
if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    tf.test.main()
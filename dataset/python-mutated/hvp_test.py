"""Tests and benchmarks for Hessian-vector products with ResNet50."""
import gc
import time
from absl.testing import parameterized
import tensorflow as tf
from tensorflow.python.eager import forwardprop
from tensorflow.python.eager.benchmarks.resnet50 import resnet50
from tensorflow.python.eager.benchmarks.resnet50 import resnet50_test_util

def _forward_over_back_hvp(model, images, labels, vector):
    if False:
        while True:
            i = 10
    with forwardprop.ForwardAccumulator(model.trainable_variables, vector) as acc:
        with tf.GradientTape() as grad_tape:
            logits = model(images, training=True)
            loss = tf.compat.v1.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
        grads = grad_tape.gradient(loss, model.trainable_variables)
    return acc.jvp(grads)

def _back_over_forward_hvp(model, images, labels, vector):
    if False:
        while True:
            i = 10
    with tf.GradientTape() as grad_tape:
        grad_tape.watch(model.trainable_variables)
        with forwardprop.ForwardAccumulator(model.trainable_variables, vector) as acc:
            logits = model(images, training=True)
            loss = tf.compat.v1.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    return grad_tape.gradient(acc.jvp(loss), model.trainable_variables)

def _tf_gradients_forward_over_back_hvp(model, images, labels, vector):
    if False:
        for i in range(10):
            print('nop')
    with tf.GradientTape() as grad_tape:
        logits = model(images, training=True)
        loss = tf.compat.v1.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    variables = model.trainable_variables
    grads = grad_tape.gradient(loss, variables)
    helpers = tf.nest.map_structure(tf.ones_like, grads)
    transposing = tf.gradients(grads, variables, helpers)
    return tf.gradients(transposing, helpers, vector)

def _back_over_back_hvp(model, images, labels, vector):
    if False:
        print('Hello World!')
    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            logits = model(images, training=True)
            loss = tf.compat.v1.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
        grads = inner_tape.gradient(loss, model.trainable_variables)
    return outer_tape.gradient(grads, model.trainable_variables, output_gradients=vector)

class HVPTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(('forward_over_back_eager', _forward_over_back_hvp), ('forward_over_back_function', tf.function(_forward_over_back_hvp)), ('tf_gradients', tf.function(_tf_gradients_forward_over_back_hvp)), ('back_over_back_eager', _back_over_back_hvp), ('back_over_back_function', tf.function(_back_over_back_hvp)), ('back_over_forward_eager', _back_over_forward_hvp), ('back_over_forward_function', tf.function(_back_over_forward_hvp)))
    def test_hvp_shapes(self, hvp_function):
        if False:
            for i in range(10):
                print('nop')
        (device, data_format) = resnet50_test_util.device_and_data_format()
        model = resnet50.ResNet50(data_format)
        with tf.device(device):
            (images, labels) = resnet50_test_util.random_batch(2, data_format)
            images = tf.constant(images)
            labels = tf.constant(labels)
            model.build(images.shape)
            vector = [tf.ones_like(v) for v in model.trainable_variables]
            hvp = hvp_function(model, images, labels, vector)
            for (hvp_component, variable) in zip(hvp, model.trainable_variables):
                self.assertEqual(hvp_component.shape, variable.shape)
                self.assertEqual(hvp_component.dtype, variable.dtype)

class HVPBenchmarks(tf.test.Benchmark):

    def _force_device_sync(self):
        if False:
            while True:
                i = 10
        tf.constant(1.0).cpu()

    def _hvp_benchmark(self, hvp_fn, label, batch_sizes, num_iters=30, num_burn=5):
        if False:
            i = 10
            return i + 15
        (device, data_format) = resnet50_test_util.device_and_data_format()
        model = resnet50.ResNet50(data_format)
        for batch_size in batch_sizes:
            with tf.device(device):
                (images, labels) = resnet50_test_util.random_batch(batch_size, data_format)
                images = tf.constant(images)
                labels = tf.constant(labels)
                model.build(images.shape)
                vector = [tf.ones_like(v) for v in model.trainable_variables]
                for _ in range(num_burn):
                    results = hvp_fn(model, images, labels, vector)
                    for result in results:
                        result.cpu()
                self._force_device_sync()
                gc.collect()
                start = time.time()
                for _ in range(num_iters):
                    results = hvp_fn(model, images, labels, vector)
                    for result in results:
                        result.cpu()
                self._force_device_sync()
                resnet50_test_util.report(self, label, start, num_iters, device, batch_size, data_format)

    def benchmark_forward_over_backward_hvp_eager(self):
        if False:
            print('Hello World!')
        self._hvp_benchmark(_forward_over_back_hvp, 'forward_over_backward_hvp_eager', batch_sizes=[8])

    def benchmark_forward_over_backward_hvp_function(self):
        if False:
            while True:
                i = 10
        self._hvp_benchmark(tf.function(_forward_over_back_hvp), 'forward_over_backward_hvp_function', batch_sizes=[8])

    def benchmark_tf_gradients_forward_over_backward_hvp_function(self):
        if False:
            i = 10
            return i + 15
        self._hvp_benchmark(tf.function(_tf_gradients_forward_over_back_hvp), 'tf_gradients_forward_over_backward_hvp_function', batch_sizes=[8])

    def benchmark_backward_over_backward_hvp_eager(self):
        if False:
            i = 10
            return i + 15
        self._hvp_benchmark(_back_over_back_hvp, 'backward_over_backward_hvp_eager', batch_sizes=[8])

    def benchmark_backward_over_backward_hvp_function(self):
        if False:
            i = 10
            return i + 15
        self._hvp_benchmark(tf.function(_back_over_back_hvp), 'backward_over_backward_hvp_function', batch_sizes=[8])

    def benchmark_backward_over_forward_hvp_eager(self):
        if False:
            print('Hello World!')
        self._hvp_benchmark(_back_over_forward_hvp, 'backward_over_forward_hvp_eager', batch_sizes=[8])

    def benchmark_backward_over_forward_hvp_function(self):
        if False:
            i = 10
            return i + 15
        self._hvp_benchmark(tf.function(_back_over_forward_hvp), 'backward_over_forward_hvp_function', batch_sizes=[8])
if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()
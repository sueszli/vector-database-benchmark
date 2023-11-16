from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import unittest
import tensorflow as tf
from official.r1.mnist import mnist
from official.utils.misc import keras_utils
BATCH_SIZE = 100

def dummy_input_fn():
    if False:
        i = 10
        return i + 15
    image = tf.random.uniform([BATCH_SIZE, 784])
    labels = tf.random.uniform([BATCH_SIZE, 1], maxval=9, dtype=tf.int32)
    return (image, labels)

def make_estimator():
    if False:
        while True:
            i = 10
    data_format = 'channels_last'
    if tf.test.is_built_with_cuda():
        data_format = 'channels_first'
    return tf.estimator.Estimator(model_fn=mnist.model_fn, params={'data_format': data_format})

class Tests(tf.test.TestCase):
    """Run tests for MNIST model.

  MNIST uses contrib and will not work with TF 2.0.  All tests are disabled if
  using TF 2.0.
  """

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def test_mnist(self):
        if False:
            i = 10
            return i + 15
        classifier = make_estimator()
        classifier.train(input_fn=dummy_input_fn, steps=2)
        eval_results = classifier.evaluate(input_fn=dummy_input_fn, steps=1)
        loss = eval_results['loss']
        global_step = eval_results['global_step']
        accuracy = eval_results['accuracy']
        self.assertEqual(loss.shape, ())
        self.assertEqual(2, global_step)
        self.assertEqual(accuracy.shape, ())
        input_fn = lambda : tf.random.uniform([3, 784])
        predictions_generator = classifier.predict(input_fn)
        for _ in range(3):
            predictions = next(predictions_generator)
            self.assertEqual(predictions['probabilities'].shape, (10,))
            self.assertEqual(predictions['classes'].shape, ())

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def mnist_model_fn_helper(self, mode, multi_gpu=False):
        if False:
            i = 10
            return i + 15
        (features, labels) = dummy_input_fn()
        image_count = features.shape[0]
        spec = mnist.model_fn(features, labels, mode, {'data_format': 'channels_last', 'multi_gpu': multi_gpu})
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = spec.predictions
            self.assertAllEqual(predictions['probabilities'].shape, (image_count, 10))
            self.assertEqual(predictions['probabilities'].dtype, tf.float32)
            self.assertAllEqual(predictions['classes'].shape, (image_count,))
            self.assertEqual(predictions['classes'].dtype, tf.int64)
        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = spec.loss
            self.assertAllEqual(loss.shape, ())
            self.assertEqual(loss.dtype, tf.float32)
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = spec.eval_metric_ops
            self.assertAllEqual(eval_metric_ops['accuracy'][0].shape, ())
            self.assertAllEqual(eval_metric_ops['accuracy'][1].shape, ())
            self.assertEqual(eval_metric_ops['accuracy'][0].dtype, tf.float32)
            self.assertEqual(eval_metric_ops['accuracy'][1].dtype, tf.float32)

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def test_mnist_model_fn_train_mode(self):
        if False:
            print('Hello World!')
        self.mnist_model_fn_helper(tf.estimator.ModeKeys.TRAIN)

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def test_mnist_model_fn_train_mode_multi_gpu(self):
        if False:
            i = 10
            return i + 15
        self.mnist_model_fn_helper(tf.estimator.ModeKeys.TRAIN, multi_gpu=True)

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def test_mnist_model_fn_eval_mode(self):
        if False:
            for i in range(10):
                print('nop')
        self.mnist_model_fn_helper(tf.estimator.ModeKeys.EVAL)

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def test_mnist_model_fn_predict_mode(self):
        if False:
            for i in range(10):
                print('nop')
        self.mnist_model_fn_helper(tf.estimator.ModeKeys.PREDICT)

class Benchmarks(tf.test.Benchmark):
    """Simple speed benchmarking for MNIST."""

    def benchmark_train_step_time(self):
        if False:
            return 10
        classifier = make_estimator()
        classifier.train(input_fn=dummy_input_fn, steps=1)
        have_gpu = tf.test.is_gpu_available()
        num_steps = 1000 if have_gpu else 100
        name = 'train_step_time_%s' % ('gpu' if have_gpu else 'cpu')
        start = time.time()
        classifier.train(input_fn=dummy_input_fn, steps=num_steps)
        end = time.time()
        wall_time = (end - start) / num_steps
        self.report_benchmark(iters=num_steps, wall_time=wall_time, name=name, extras={'examples_per_sec': BATCH_SIZE / wall_time})
if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.test.main()
"""Tests for object_detection.core.balanced_positive_negative_sampler."""
import numpy as np
import tensorflow as tf
from object_detection.core import balanced_positive_negative_sampler
from object_detection.utils import test_case

class BalancedPositiveNegativeSamplerTest(test_case.TestCase):

    def test_subsample_all_examples_dynamic(self):
        if False:
            i = 10
            return i + 15
        numpy_labels = np.random.permutation(300)
        indicator = tf.constant(np.ones(300) == 1)
        numpy_labels = numpy_labels - 200 > 0
        labels = tf.constant(numpy_labels)
        sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler()
        is_sampled = sampler.subsample(indicator, 64, labels)
        with self.test_session() as sess:
            is_sampled = sess.run(is_sampled)
            self.assertTrue(sum(is_sampled) == 64)
            self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 32)
            self.assertTrue(sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) == 32)

    def test_subsample_all_examples_static(self):
        if False:
            i = 10
            return i + 15
        numpy_labels = np.random.permutation(300)
        indicator = np.array(np.ones(300) == 1, np.bool)
        numpy_labels = numpy_labels - 200 > 0
        labels = np.array(numpy_labels, np.bool)

        def graph_fn(indicator, labels):
            if False:
                print('Hello World!')
            sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(is_static=True)
            return sampler.subsample(indicator, 64, labels)
        is_sampled = self.execute(graph_fn, [indicator, labels])
        self.assertTrue(sum(is_sampled) == 64)
        self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 32)
        self.assertTrue(sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) == 32)

    def test_subsample_selection_dynamic(self):
        if False:
            while True:
                i = 10
        numpy_labels = np.arange(100)
        numpy_indicator = numpy_labels < 90
        indicator = tf.constant(numpy_indicator)
        numpy_labels = numpy_labels - 80 >= 0
        labels = tf.constant(numpy_labels)
        sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler()
        is_sampled = sampler.subsample(indicator, 64, labels)
        with self.test_session() as sess:
            is_sampled = sess.run(is_sampled)
            self.assertTrue(sum(is_sampled) == 64)
            self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 10)
            self.assertTrue(sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) == 54)
            self.assertAllEqual(is_sampled, np.logical_and(is_sampled, numpy_indicator))

    def test_subsample_selection_static(self):
        if False:
            for i in range(10):
                print('nop')
        numpy_labels = np.arange(100)
        numpy_indicator = numpy_labels < 90
        indicator = np.array(numpy_indicator, np.bool)
        numpy_labels = numpy_labels - 80 >= 0
        labels = np.array(numpy_labels, np.bool)

        def graph_fn(indicator, labels):
            if False:
                print('Hello World!')
            sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(is_static=True)
            return sampler.subsample(indicator, 64, labels)
        is_sampled = self.execute(graph_fn, [indicator, labels])
        self.assertTrue(sum(is_sampled) == 64)
        self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 10)
        self.assertTrue(sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) == 54)
        self.assertAllEqual(is_sampled, np.logical_and(is_sampled, numpy_indicator))

    def test_subsample_selection_larger_batch_size_dynamic(self):
        if False:
            while True:
                i = 10
        numpy_labels = np.arange(100)
        numpy_indicator = numpy_labels < 60
        indicator = tf.constant(numpy_indicator)
        numpy_labels = numpy_labels - 50 >= 0
        labels = tf.constant(numpy_labels)
        sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler()
        is_sampled = sampler.subsample(indicator, 64, labels)
        with self.test_session() as sess:
            is_sampled = sess.run(is_sampled)
            self.assertTrue(sum(is_sampled) == 60)
            self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 10)
            self.assertTrue(sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) == 50)
            self.assertAllEqual(is_sampled, np.logical_and(is_sampled, numpy_indicator))

    def test_subsample_selection_larger_batch_size_static(self):
        if False:
            print('Hello World!')
        numpy_labels = np.arange(100)
        numpy_indicator = numpy_labels < 60
        indicator = np.array(numpy_indicator, np.bool)
        numpy_labels = numpy_labels - 50 >= 0
        labels = np.array(numpy_labels, np.bool)

        def graph_fn(indicator, labels):
            if False:
                i = 10
                return i + 15
            sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(is_static=True)
            return sampler.subsample(indicator, 64, labels)
        is_sampled = self.execute(graph_fn, [indicator, labels])
        self.assertTrue(sum(is_sampled) == 64)
        self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) >= 10)
        self.assertTrue(sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) >= 50)
        self.assertTrue(sum(np.logical_and(is_sampled, numpy_indicator)) == 60)

    def test_subsample_selection_no_batch_size(self):
        if False:
            i = 10
            return i + 15
        numpy_labels = np.arange(1000)
        numpy_indicator = numpy_labels < 999
        indicator = tf.constant(numpy_indicator)
        numpy_labels = numpy_labels - 994 >= 0
        labels = tf.constant(numpy_labels)
        sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(0.01)
        is_sampled = sampler.subsample(indicator, None, labels)
        with self.test_session() as sess:
            is_sampled = sess.run(is_sampled)
            self.assertTrue(sum(is_sampled) == 500)
            self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 5)
            self.assertTrue(sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) == 495)
            self.assertAllEqual(is_sampled, np.logical_and(is_sampled, numpy_indicator))

    def test_subsample_selection_no_batch_size_static(self):
        if False:
            return 10
        labels = tf.constant([[True, False, False]])
        indicator = tf.constant([True, False, True])
        sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler()
        with self.assertRaises(ValueError):
            sampler.subsample(indicator, None, labels)

    def test_raises_error_with_incorrect_label_shape(self):
        if False:
            return 10
        labels = tf.constant([[True, False, False]])
        indicator = tf.constant([True, False, True])
        sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler()
        with self.assertRaises(ValueError):
            sampler.subsample(indicator, 64, labels)

    def test_raises_error_with_incorrect_indicator_shape(self):
        if False:
            i = 10
            return i + 15
        labels = tf.constant([True, False, False])
        indicator = tf.constant([[True, False, True]])
        sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler()
        with self.assertRaises(ValueError):
            sampler.subsample(indicator, 64, labels)
if __name__ == '__main__':
    tf.test.main()
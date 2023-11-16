"""Tests and benchmarks for ResNet50 under graph execution."""
import time
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.eager.benchmarks.resnet50 import resnet50

def data_format():
    if False:
        print('Hello World!')
    return 'channels_first' if tf.test.is_gpu_available() else 'channels_last'

def image_shape(batch_size):
    if False:
        for i in range(10):
            print('nop')
    if data_format() == 'channels_first':
        return [batch_size, 3, 224, 224]
    return [batch_size, 224, 224, 3]

def random_batch(batch_size):
    if False:
        print('Hello World!')
    images = np.random.rand(*image_shape(batch_size)).astype(np.float32)
    num_classes = 1000
    labels = np.random.randint(low=0, high=num_classes, size=[batch_size]).astype(np.int32)
    one_hot = np.zeros((batch_size, num_classes)).astype(np.float32)
    one_hot[np.arange(batch_size), labels] = 1.0
    return (images, one_hot)

class ResNet50GraphTest(tf.test.TestCase):

    def testApply(self):
        if False:
            print('Hello World!')
        batch_size = 8
        with tf.Graph().as_default():
            images = tf.placeholder(tf.float32, image_shape(None))
            model = resnet50.ResNet50(data_format())
            predictions = model(images, training=False)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                (np_images, _) = random_batch(batch_size)
                out = sess.run(predictions, feed_dict={images: np_images})
                self.assertAllEqual([batch_size, 1000], out.shape)

class ResNet50Benchmarks(tf.test.Benchmark):

    def _report(self, label, start, num_iters, batch_size):
        if False:
            for i in range(10):
                print('nop')
        avg_time = (time.time() - start) / num_iters
        dev = 'gpu' if tf.test.is_gpu_available() else 'cpu'
        name = 'graph_%s_%s_batch_%d_%s' % (label, dev, batch_size, data_format())
        extras = {'examples_per_sec': batch_size / avg_time}
        self.report_benchmark(iters=num_iters, wall_time=avg_time, name=name, extras=extras)

    def benchmark_graph_apply(self):
        if False:
            i = 10
            return i + 15
        with tf.Graph().as_default():
            images = tf.placeholder(tf.float32, image_shape(None))
            model = resnet50.ResNet50(data_format())
            predictions = model(images, training=False)
            init = tf.global_variables_initializer()
            batch_size = 64
            with tf.Session() as sess:
                sess.run(init)
                (np_images, _) = random_batch(batch_size)
                (num_burn, num_iters) = (3, 30)
                for _ in range(num_burn):
                    sess.run(predictions, feed_dict={images: np_images})
                start = time.time()
                for _ in range(num_iters):
                    sess.run(predictions, feed_dict={images: np_images})
                self._report('apply', start, num_iters, batch_size)

    def benchmark_graph_train(self):
        if False:
            return 10
        for batch_size in [16, 32, 64]:
            with tf.Graph().as_default():
                (np_images, np_labels) = random_batch(batch_size)
                dataset = tf.data.Dataset.from_tensors((np_images, np_labels)).repeat()
                (images, labels) = tf.data.make_one_shot_iterator(dataset).get_next()
                model = resnet50.ResNet50(data_format())
                logits = model(images, training=True)
                loss = tf.compat.v1.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
                train_op = optimizer.minimize(loss)
                init = tf.global_variables_initializer()
                with tf.Session() as sess:
                    sess.run(init)
                    (num_burn, num_iters) = (5, 10)
                    for _ in range(num_burn):
                        sess.run(train_op)
                    start = time.time()
                    for _ in range(num_iters):
                        sess.run(train_op)
                    self._report('train', start, num_iters, batch_size)
if __name__ == '__main__':
    tf.test.main()
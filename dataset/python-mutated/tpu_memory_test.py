"""This file contains integration test for TPUStrategy in regards to memory."""
import gc
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
NUM_CLASS = 10

def get_dataset():
    if False:
        return 10

    def generate_data(_):
        if False:
            print('Hello World!')
        image = tf.ones([500, 500, 3], dtype=tf.float32)
        label = tf.zeros([1], dtype=tf.int32)
        return (image, label)

    def preprocess(image, label):
        if False:
            while True:
                i = 10
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, NUM_CLASS)
        label = tf.reshape(label, [NUM_CLASS])
        return (image, label)
    dataset = tf.data.Dataset.range(1)
    dataset = dataset.repeat()
    dataset = dataset.map(generate_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(128, drop_remainder=True)
    return dataset

class TpuMemoryTest(tf.test.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        context._reset_context()
        gc.collect()
        assert tf.reduce_sum(tf.random.uniform((1024, 128), dtype=tf.float32)).numpy() > 1.0
        self.resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='', project=None, zone=None)
        tf.config.experimental_connect_to_cluster(self.resolver)
        tf.tpu.experimental.initialize_tpu_system(self.resolver)

    def testAutoDefragInProgramLoading(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = tf.distribute.TPUStrategy(self.resolver)
        dataset = get_dataset()
        iterator = iter(strategy.experimental_distribute_dataset(dataset, tf.distribute.InputOptions()))
        with strategy.scope():
            x = tf.keras.layers.Input(shape=(500, 500, 3), name='input')
            y = tf.keras.layers.Conv2D(384, (15, 15), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name='conv1')(x)
            y = tf.keras.layers.BatchNormalization(momentum=0.997, center=True, scale=True)(y)
            y = tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(y)
            y = tf.keras.layers.Conv2D(64, (9, 9), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal', name='conv2')(y)
            y = tf.keras.layers.Flatten()(y)
            y = tf.keras.layers.Dense(1024, activation='softmax', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(y)
            y = tf.keras.layers.Dense(1024, activation='softmax', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(y)
            y = tf.keras.layers.Dense(NUM_CLASS, activation='softmax', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(y)
            model = tf.keras.Model(x, y)
            optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.1)
            loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0, reduction=tf.keras.losses.Reduction.NONE)
            model.compile(optimizer=optimizer, loss=loss_obj)

        @tf.function
        def train_step(iterator):
            if False:
                return 10

            def step_fn(inputs):
                if False:
                    for i in range(10):
                        print('nop')
                (images, targets) = inputs
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    loss = model.loss(targets, outputs)
                grads = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                return loss
            for _ in tf.range(tf.constant(20)):
                strategy.run(step_fn, args=(next(iterator),))
            strategy.run(step_fn, args=(next(iterator),))
            return 1.0
        if FLAGS.tpu_use_tfrt:
            result = train_step(iterator)
            self.assertAllClose(1.0, result, atol=1e-07)
        else:
            with self.assertRaises(tf.errors.ResourceExhaustedError):
                _ = train_step(iterator)

    def testAutoDefragInBufferAllocation(self):
        if False:
            print('Hello World!')
        if not FLAGS.tpu_use_tfrt:
            self.skipTest('TPU StreamExecutor does not support auto-defrag in allocation.')
        with tf.device('TPU:0'):
            buffer_2g_1 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
            buffer_2g_2 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
            buffer_2g_3 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
            buffer_2g_4 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
            buffer_2g_5 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
            buffer_2g_6 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
            buffer_2g_7 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
            del buffer_2g_1, buffer_2g_3
            gc.collect()
            buffer_4g = tf.random.uniform((4, 256, 1024, 1024), dtype=tf.float32)
        self.assertEndsWith(buffer_4g.device, 'device:TPU:0')
if __name__ == '__main__':
    tf.test.main()
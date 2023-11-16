import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from bigdl.nano.tf.keras import Sequential, nano_bf16

def test_tf_nano_bf16_decorator():
    if False:
        i = 10
        return i + 15
    from bigdl.nano.tf import patch_tensorflow, unpatch_tensorflow
    patch_tensorflow(precision='mixed_bfloat16')

    class Model:
        model = Sequential([layers.Dense(units=1, input_shape=[1])])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        def loss(self, y, pred):
            if False:
                while True:
                    i = 10
            return tf.losses.mean_squared_error(y, pred)

        @nano_bf16
        @tf.function
        def train(self, x, y):
            if False:
                i = 10
                return i + 15
            with tf.GradientTape() as tape:
                pred = self.model(x, training=True)
                loss_value = self.loss(y, pred)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss_value
    model = Model()
    unpatch_tensorflow()

def test_tf_nano_multiprocessing_customized_loop():
    if False:
        return 10
    from bigdl.nano.tf.keras import nano_multiprocessing, nano
    global_batch_size = 32
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    optimizer = tf.keras.optimizers.SGD()
    dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(128).batch(global_batch_size)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @nano_multiprocessing
    @tf.function
    def train_step(inputs, model, loss_object, optimizer):
        if False:
            for i in range(10):
                print('nop')
        (features, labels) = inputs
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @nano(num_processes=2)
    def train_whole_data(model, dataset, loss_object, optimizer, train_step):
        if False:
            while True:
                i = 10
        for inputs in dataset:
            print(train_step(inputs, model, loss_object, optimizer))
    train_whole_data(model, dataset, loss_object, optimizer, train_step)

def test_tf_nano_multiprocessing_customized_loss_datagenerator():
    if False:
        for i in range(10):
            print('nop')
    from bigdl.nano.tf.keras import nano_multiprocessing, nano, nano_multiprocessing_loss
    global_batch_size = 32
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    optimizer = tf.keras.optimizers.SGD()

    def dummy_data_generator():
        if False:
            for i in range(10):
                print('nop')
        for i in range(128):
            yield (tf.constant([i]), tf.constant([i]))
    dataset = tf.data.Dataset.from_generator(dummy_data_generator, output_signature=(tf.TensorSpec(shape=(1,), dtype=tf.float32), tf.TensorSpec(shape=(1,), dtype=tf.float32)))
    dataset._GeneratorState = dataset._GeneratorState(dummy_data_generator)

    @nano_multiprocessing_loss()
    def loss_object(x, pred):
        if False:
            return 10
        res = backend.mean(tf.math.squared_difference(x, pred), axis=-1)
        return res

    @nano_multiprocessing
    @tf.function
    def train_step(inputs, model, loss_object, optimizer):
        if False:
            return 10
        (features, labels) = inputs
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @nano(num_processes=2)
    def train_whole_data(model, dataset, loss_object, optimizer, train_step):
        if False:
            return 10
        for inputs in dataset:
            print(train_step(inputs, model, loss_object, optimizer))
    train_whole_data(model, dataset, loss_object, optimizer, train_step)
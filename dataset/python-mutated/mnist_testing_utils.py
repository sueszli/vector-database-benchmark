"""A simple MNIST model for testing multi-worker distribution strategies with Keras."""
import tensorflow as tf

def mnist_synthetic_dataset(batch_size, steps_per_epoch):
    if False:
        i = 10
        return i + 15
    'Generate synthetic MNIST dataset for testing.'
    x_train = tf.ones([batch_size * steps_per_epoch, 28, 28, 1], dtype=tf.dtypes.float32)
    y_train = tf.ones([batch_size * steps_per_epoch, 1], dtype=tf.dtypes.int32)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(64, drop_remainder=True)
    x_test = tf.random.uniform([10000, 28, 28, 1], dtype=tf.dtypes.float32)
    y_test = tf.random.uniform([10000, 1], minval=0, maxval=9, dtype=tf.dtypes.int32)
    eval_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    eval_ds = eval_ds.batch(64, drop_remainder=True)
    return (train_ds, eval_ds)

def get_mnist_model(input_shape):
    if False:
        print('Hello World!')
    'Define a deterministically-initialized CNN model for MNIST testing.'
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.TruncatedNormal(seed=99))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x) + tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.TruncatedNormal(seed=99))(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
    return model
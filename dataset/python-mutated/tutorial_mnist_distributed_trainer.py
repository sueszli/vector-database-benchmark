import numpy as np
import tensorflow as tf
import tensorlayer as tl
tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

def make_dataset(images, labels, num_epochs=1, shuffle_data_seed=0):
    if False:
        return 10
    ds1 = tf.data.Dataset.from_tensor_slices(images)
    ds2 = tf.data.Dataset.from_tensor_slices(np.array(labels, dtype=np.int64))
    dataset = tf.data.Dataset.zip((ds1, ds2))
    dataset = dataset.repeat(num_epochs).shuffle(buffer_size=10000, seed=shuffle_data_seed)
    return dataset

def model(x, is_train):
    if False:
        i = 10
        return i + 15
    with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
        network = tl.layers.InputLayer(x, name='input')
        network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1', is_fix=True, is_train=is_train)
        network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu1')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2', is_fix=True, is_train=is_train)
        network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu2')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3', is_fix=True, is_train=is_train)
        network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output')
    return network

def build_train(x, y_):
    if False:
        print('Hello World!')
    net = model(x, is_train=True)
    cost = tl.cost.cross_entropy(net.outputs, y_, name='cost_train')
    accurate_prediction = tf.equal(tf.argmax(net.outputs, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(accurate_prediction, tf.float32), name='accuracy_train')
    log_tensors = {'cost': cost, 'accuracy': accuracy}
    return (net, cost, log_tensors)

def build_validation(x, y_):
    if False:
        while True:
            i = 10
    net = model(x, is_train=False)
    cost = tl.cost.cross_entropy(net.outputs, y_, name='cost_test')
    accurate_prediction = tf.equal(tf.argmax(net.outputs, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(accurate_prediction, tf.float32), name='accuracy_test')
    return (net, [cost, accuracy])
if __name__ == '__main__':
    (X_train, y_train, X_val, y_val, X_test, y_test) = tl.files.load_mnist_dataset(shape=(-1, 784))
    training_dataset = make_dataset(X_train, y_train)
    trainer = tl.distributed.Trainer(build_training_func=build_train, training_dataset=training_dataset, optimizer=tf.train.AdamOptimizer, optimizer_args={'learning_rate': 0.001}, batch_size=500, prefetch_size=500)
    while not trainer.session.should_stop():
        try:
            trainer.train_on_batch()
        except tf.errors.OutOfRangeError:
            break
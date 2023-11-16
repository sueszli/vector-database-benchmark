import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)
((x, y), _) = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 50.0
y = tf.convert_to_tensor(y)
y = tf.one_hot(y, depth=10)
print('x:', x.shape, 'y:', y.shape)
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128).repeat(30)
(x, y) = next(iter(train_db))
print('sample:', x.shape, y.shape)

def main():
    if False:
        return 10
    (w1, b1) = (tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1)), tf.Variable(tf.zeros([512])))
    (w2, b2) = (tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1)), tf.Variable(tf.zeros([256])))
    (w3, b3) = (tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1)), tf.Variable(tf.zeros([10])))
    optimizer = optimizers.SGD(lr=0.01)
    for (step, (x, y)) in enumerate(train_db):
        x = tf.reshape(x, (-1, 784))
        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2 @ w3 + b3
            loss = tf.square(y - out)
            loss = tf.reduce_mean(loss, axis=1)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        (grads, _) = tf.clip_by_global_norm(grads, 15)
        optimizer.apply_gradients(zip(grads, [w1, b1, w2, b2, w3, b3]))
        if step % 100 == 0:
            print(step, 'loss:', float(loss))
if __name__ == '__main__':
    main()
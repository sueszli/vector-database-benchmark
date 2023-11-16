from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
original = np.sin(np.linspace(0, 3 * np.pi, 300))
X = 2 * np.random.randn(300) + original
plt.plot(X)
plt.title('original')
plt.show()
decay = tf.placeholder(tf.float32, shape=(), name='decay')
sequence = tf.placeholder(tf.float32, shape=(None,), name='sequence')

def recurrence(last, x):
    if False:
        i = 10
        return i + 15
    return (1.0 - decay) * x + decay * last
lpf = tf.scan(fn=recurrence, elems=sequence, initializer=0.0)
with tf.Session() as session:
    Y = session.run(lpf, feed_dict={sequence: X, decay: 0.97})
    plt.plot(Y)
    plt.plot(original)
    plt.title('filtered')
    plt.show()
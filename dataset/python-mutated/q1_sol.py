"""
Solution to simple exercises to get used to TensorFlow API
You should thoroughly test your code.
TensorFlow's official documentation should be your best friend here
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.random_uniform([])
y = tf.random_uniform([])
out = tf.cond(tf.greater(x, y), lambda : tf.add(x, y), lambda : tf.subtract(x, y))
x = tf.random_uniform([], -1, 1, dtype=tf.float32)
y = tf.random_uniform([], -1, 1, dtype=tf.float32)
out = tf.case({tf.less(x, y): lambda : tf.add(x, y), tf.greater(x, y): lambda : tf.subtract(x, y)}, default=lambda : tf.constant(0.0), exclusive=True)
x = tf.constant([[0, -2, -1], [0, 1, 2]])
y = tf.zeros_like(x)
out = tf.equal(x, y)
x = tf.constant([29.05088806, 27.61298943, 31.19073486, 29.35532951, 30.97266006, 26.67541885, 38.08450317, 20.74983215, 34.94445419, 34.45999146, 29.06485367, 36.01657104, 27.88236427, 20.56035233, 30.20379066, 29.51215172, 33.71149445, 28.59134293, 36.05556488, 28.66994858])
indices = tf.where(x > 30)
out = tf.gather(x, indices)
values = tf.range(1, 7)
out = tf.diag(values)
m = tf.random_normal([10, 10], mean=10, stddev=1)
out = tf.matrix_determinant(m)
x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
(unique_values, indices) = tf.unique(x)
x = tf.random_normal([300], mean=5, stddev=1)
y = tf.random_normal([300], mean=5, stddev=1)
average = tf.reduce_mean(x - y)

def f1():
    if False:
        print('Hello World!')
    return tf.reduce_mean(tf.square(x - y))

def f2():
    if False:
        while True:
            i = 10
    return tf.reduce_sum(tf.abs(x - y))
out = tf.cond(average < 0, f1, f2)
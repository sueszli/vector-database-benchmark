from __future__ import print_function, division
from builtins import range
import numpy as np
import tensorflow as tf
x = tf.placeholder(tf.int32, shape=(None,), name='x')

def square(last, current):
    if False:
        for i in range(10):
            print('nop')
    return current * current
square_op = tf.scan(fn=square, elems=x)
with tf.Session() as session:
    o_val = session.run(square_op, feed_dict={x: [1, 2, 3, 4, 5]})
    print('output:', o_val)
from __future__ import print_function, division
from builtins import range
import numpy as np
import tensorflow as tf
N = tf.placeholder(tf.int32, shape=(), name='N')

def recurrence(last_output, current_input):
    if False:
        print('Hello World!')
    return (last_output[1], last_output[0] + last_output[1])
fibonacci = tf.scan(fn=recurrence, elems=tf.range(N), initializer=(0, 1))
with tf.Session() as session:
    o_val = session.run(fibonacci, feed_dict={N: 8})
    print('output:', o_val)
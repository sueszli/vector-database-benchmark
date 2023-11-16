"""Exports a toy TensorFlow model without signatures.

Exports half_plus_two TensorFlow model to /tmp/bad_half_plus_two/ without
signatures. This is used to test the fault-tolerance of tensorflow_model_server.
"""
import os
import tensorflow as tf

def Export():
    if False:
        return 10
    export_path = '/tmp/bad_half_plus_two/00000123'
    with tf.Session() as sess:
        a = tf.Variable(0.5)
        b = tf.Variable(2.0)
        x = tf.placeholder(tf.float32)
        y = tf.add(tf.multiply(a, x), b)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.export_meta_graph(filename=os.path.join(export_path, 'export.meta'))
        saver.save(sess, os.path.join(export_path, 'export'), write_meta_graph=False)

def main(_):
    if False:
        i = 10
        return i + 15
    Export()
if __name__ == '__main__':
    tf.compat.v1.app.run()
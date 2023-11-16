"""KL-divergence metrics calculation."""
import tensorflow.compat.v1 as tf

def symmetric_kl_divergence(predicted, actual):
    if False:
        while True:
            i = 10
    'Calculate symmetric KL-divergence over two classification tensors.\n\n  Note that here the classifications do not form a probability distribution.\n  They are, however normalized to 0..1 and calculating a KL-divergence over them\n  gives reasonable numerical results.\n\n  Shape of the two inputs must be the same at inference time but is otherwise\n  unconstrained.\n\n  Args:\n    predicted: classification outputs from model\n    actual: golden classification outputs\n\n  Returns:\n    Single scalar tensor with symmetric KL-divergence between predicted and\n    actual.\n  '
    epsilon = tf.constant(1e-07, dtype=tf.float32, name='epsilon')
    p = tf.math.maximum(predicted, epsilon)
    q = tf.math.maximum(actual, epsilon)
    kld_1 = tf.math.reduce_sum(tf.math.multiply(p, tf.math.log(tf.math.divide(p, q))))
    kld_2 = tf.math.reduce_sum(tf.math.multiply(q, tf.math.log(tf.math.divide(q, p))))
    return tf.add(kld_1, kld_2)
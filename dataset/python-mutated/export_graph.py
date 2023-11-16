"""An example of building and exporting a Tensorflow graph.

Adapted from the Travis Ebesu's blog post:
https://tebesu.github.io/posts/Training-a-TensorFlow-graph-in-C++-API
"""
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
import pyspiel
FLAGS = flags.FLAGS
flags.DEFINE_string('game', 'breakthrough', 'Name of the game')
flags.DEFINE_string('dir', '/tmp', 'Directory to save graph')
flags.DEFINE_string('filename', 'graph.pb', 'Filename for the graph')

def main(_):
    if False:
        return 10
    game = pyspiel.load_game(FLAGS.game)
    info_state_shape = game.observation_tensor_shape()
    flat_info_state_length = np.prod(info_state_shape)
    num_actions = game.num_distinct_actions()
    with tf.Session() as sess:
        net_input = tf.placeholder(tf.float32, [None, flat_info_state_length], name='input')
        output = tf.placeholder(tf.float32, [None, num_actions], name='output')
        legals_mask = tf.placeholder(tf.float32, [None, num_actions], name='legals_mask')
        policy_net = tf.layers.dense(net_input, 128, activation=tf.nn.relu)
        policy_net = tf.layers.dense(policy_net, 128, activation=tf.nn.relu)
        policy_net = tf.layers.dense(policy_net, num_actions)
        policy_net = policy_net - tf.reduce_max(policy_net, axis=-1, keepdims=True)
        masked_exp_logit = tf.multiply(tf.exp(policy_net), legals_mask)
        renormalizing_factor = tf.reduce_sum(masked_exp_logit, axis=-1, keepdims=True)
        policy_softmax = tf.where(tf.equal(legals_mask, 0.0), tf.zeros_like(masked_exp_logit), tf.divide(masked_exp_logit, renormalizing_factor), name='policy_softmax')
        policy_targets = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)
        policy_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=policy_net, labels=policy_targets), axis=0)
        sampled_actions = tf.random.categorical(tf.log(policy_softmax), 1, name='sampled_actions')
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(policy_cost, name='train')
        init = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
        print('Writing file: {}/{}'.format(FLAGS.dir, FLAGS.filename))
        tf.train.write_graph(sess.graph_def, FLAGS.dir, FLAGS.filename, as_text=False)
if __name__ == '__main__':
    app.run(main)
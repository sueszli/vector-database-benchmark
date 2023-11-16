"""Variational Dropout."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def generate_dropout_masks(keep_prob, shape, amount):
    if False:
        return 10
    masks = []
    for _ in range(amount):
        dropout_mask = tf.random_uniform(shape) + keep_prob
        dropout_mask = tf.floor(dropout_mask) / keep_prob
        masks.append(dropout_mask)
    return masks

def generate_variational_dropout_masks(hparams, keep_prob):
    if False:
        i = 10
        return i + 15
    [batch_size, num_steps, size, num_layers] = [FLAGS.batch_size, FLAGS.sequence_length, hparams.gen_rnn_size, hparams.gen_num_layers]
    if len(keep_prob) == 2:
        emb_keep_prob = keep_prob[0]
        h2h_keep_prob = emb_keep_prob
        h2i_keep_prob = keep_prob[1]
        out_keep_prob = h2i_keep_prob
    else:
        emb_keep_prob = keep_prob[0]
        h2h_keep_prob = keep_prob[1]
        h2i_keep_prob = keep_prob[2]
        out_keep_prob = keep_prob[3]
    h2i_masks = []
    h2h_masks = []
    emb_masks = generate_dropout_masks(emb_keep_prob, [num_steps, 1], batch_size)
    output_mask = generate_dropout_masks(out_keep_prob, [batch_size, size], 1)[0]
    h2i_masks = generate_dropout_masks(h2i_keep_prob, [batch_size, size], num_layers)
    h2h_masks = generate_dropout_masks(h2h_keep_prob, [batch_size, size], num_layers)
    return (h2h_masks, h2i_masks, emb_masks, output_mask)
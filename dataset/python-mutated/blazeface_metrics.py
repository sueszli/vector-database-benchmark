"""Metrics model generator for Blazeface.

The produced model is to be used as part of the mini-benchmark, combined into
the same flatbuffer with the main model.

The blazeface model is described in
https://tfhub.dev/tensorflow/tfjs-model/blazeface/1/default/1

The metrics are roughly equivalent to the training time loss function for SSD
(https://arxiv.org/abs/1512.02325): localization loss and classification loss.

The localization loss is MSE (L2-norm) of box encodings over high-probability
boxes. A box encoding contains the size and location difference between the
prediction and the prototype box (see section 2 in the linked paper).

The classification loss is symmetric KL-divergence over classification scores
squashed to 0..1.

This follows the general rationale of the mini-benchmark: use as much of the
model outputs as possible for metrics, so that less example data is needed.
"""
import argparse
import sys
import tensorflow.compat.v1 as tf
from tensorflow.lite.experimental.acceleration.mini_benchmark.metrics import kl_divergence
from tensorflow.lite.tools import flatbuffer_utils
parser = argparse.ArgumentParser(description='Script to generate a metrics model for the Blazeface.')
parser.add_argument('output', help='Output filepath')

@tf.function
def metrics(expected_box_encodings, expected_scores, actual_box_encodings, actual_scores):
    if False:
        for i in range(10):
            print('nop')
    'Calculate metrics from expected and actual blazeface outputs.\n\n  Args:\n    expected_box_encodings: box encodings from model\n    expected_scores: classifications from model\n    actual_box_encodings: golden box encodings\n    actual_scores: golden classifications\n\n  Returns:\n    two-item list with classification error and localization error\n  '
    squashed_expected_scores = tf.math.divide(1.0, 1.0 + tf.math.exp(-expected_scores))
    squashed_actual_scores = tf.math.divide(1.0, 1.0 + tf.math.exp(-actual_scores))
    kld_metric = kl_divergence.symmetric_kl_divergence(expected_scores, actual_scores)
    high_scoring_indices = tf.math.logical_or(tf.math.greater(squashed_expected_scores, 0.1), tf.math.greater(squashed_actual_scores, 0.1))
    high_scoring_actual_boxes = tf.where(condition=tf.broadcast_to(input=high_scoring_indices, shape=tf.shape(actual_box_encodings)), x=actual_box_encodings, y=expected_box_encodings)
    box_diff = high_scoring_actual_boxes - expected_box_encodings
    box_squared_diff = tf.math.pow(box_diff, 2)
    box_mse = tf.divide(tf.math.reduce_sum(box_squared_diff), tf.math.maximum(tf.math.count_nonzero(high_scoring_indices, dtype=tf.float32), 1.0))
    ok = tf.logical_and(kld_metric < 0.1, box_mse < 0.01)
    return [kld_metric, box_mse, ok]

def main(output_path):
    if False:
        return 10
    tf.reset_default_graph()
    with tf.Graph().as_default():
        expected_box_encodings = tf.placeholder(dtype=tf.float32, shape=[1, 564, 16])
        expected_scores = tf.placeholder(dtype=tf.float32, shape=[1, 564, 1])
        actual_box_encodings = tf.placeholder(dtype=tf.float32, shape=[1, 564, 16])
        actual_scores = tf.placeholder(dtype=tf.float32, shape=[1, 564, 1])
        [kld_metric, box_mse, ok] = metrics(expected_box_encodings, expected_scores, actual_box_encodings, actual_scores)
        ok = tf.reshape(ok, [1], name='ok')
        kld_metric = tf.reshape(kld_metric, [1], name='symmetric_kl_divergence')
        box_mse = tf.reshape(box_mse, [1], name='box_mse')
        sess = tf.compat.v1.Session()
        converter = tf.lite.TFLiteConverter.from_session(sess, [expected_box_encodings, expected_scores, actual_box_encodings, actual_scores], [kld_metric, box_mse, ok])
        converter.experimental_new_converter = True
        tflite_model = converter.convert()
        if sys.byteorder == 'big':
            tflite_model = flatbuffer_utils.byte_swap_tflite_buffer(tflite_model, 'big', 'little')
        open(output_path, 'wb').write(tflite_model)
if __name__ == '__main__':
    (flags, unparsed) = parser.parse_known_args()
    if unparsed:
        parser.print_usage()
        sys.stderr.write('\nGot the following unparsed args, %r please fix.\n' % unparsed)
        exit(1)
    else:
        main(flags.output)
        exit(0)
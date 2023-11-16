"""Exports a counter model.

It contains 4 signatures: get_counter incr_counter, incr_counter_by, and
reset_counter, to test Predict service.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def save_model(sess, signature_def_map, output_dir):
    if False:
        return 10
    'Saves the model with given signature def map.'
    builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map=signature_def_map)
    builder.save()

def build_signature_def_from_tensors(inputs, outputs, method_name):
    if False:
        print('Hello World!')
    'Builds signature def with inputs, outputs, and method_name.'
    return tf.saved_model.signature_def_utils.build_signature_def(inputs={key: tf.saved_model.utils.build_tensor_info(tensor) for (key, tensor) in inputs.items()}, outputs={key: tf.saved_model.utils.build_tensor_info(tensor) for (key, tensor) in outputs.items()}, method_name=method_name)

def export_model(output_dir):
    if False:
        return 10
    'Exports the counter model.\n\n  Create three signatures: incr_counter, incr_counter_by, reset_counter.\n\n  *Notes*: These signatures are stateful and over-simplied only to demonstrate\n  Predict calls with only inputs or outputs. State is not supported in\n  TensorFlow Serving on most scalable or production hosting environments.\n\n  Args:\n    output_dir: string, output directory for the model.\n  '
    tf.logging.info('Exporting the counter model to %s.', output_dir)
    method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    graph = tf.Graph()
    with graph.as_default(), tf.Session() as sess:
        counter = tf.Variable(0.0, dtype=tf.float32, name='counter')
        with tf.name_scope('incr_counter_op', values=[counter]):
            incr_counter = counter.assign_add(1.0)
        delta = tf.placeholder(dtype=tf.float32, name='delta')
        with tf.name_scope('incr_counter_by_op', values=[counter, delta]):
            incr_counter_by = counter.assign_add(delta)
        with tf.name_scope('reset_counter_op', values=[counter]):
            reset_counter = counter.assign(0.0)
        sess.run(tf.global_variables_initializer())
        signature_def_map = {'get_counter': build_signature_def_from_tensors({}, {'output': counter}, method_name), 'incr_counter': build_signature_def_from_tensors({}, {'output': incr_counter}, method_name), 'incr_counter_by': build_signature_def_from_tensors({'delta': delta}, {'output': incr_counter_by}, method_name), 'reset_counter': build_signature_def_from_tensors({}, {'output': reset_counter}, method_name)}
        save_model(sess, signature_def_map, output_dir)

def main(unused_argv):
    if False:
        while True:
            i = 10
    export_model('/tmp/saved_model_counter/00000123')
if __name__ == '__main__':
    tf.compat.v1.app.run()
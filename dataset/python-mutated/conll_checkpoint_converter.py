"""Conversion script for CoNLL checkpoints to DRAGNN SavedModel format.

This script loads and finishes a CoNLL checkpoint, then exports it as a
SavedModel. It expects that the CoNLL RNN cells have been updated using the
RNN update script.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import flags
import tensorflow as tf
from google.protobuf import text_format
from dragnn.protos import spec_pb2
from dragnn.python import dragnn_model_saver_lib as saver_lib
from dragnn.python import spec_builder
FLAGS = flags.FLAGS
flags.DEFINE_string('master_spec', None, 'Path to task context with inputs and parameters for feature extractors.')
flags.DEFINE_string('params_path', None, 'Path to trained model parameters.')
flags.DEFINE_string('export_path', '', 'Output path for exported servo model.')
flags.DEFINE_string('resource_path', '', 'Base directory for resources in the master spec.')
flags.DEFINE_bool('export_moving_averages', True, 'Whether to export the moving average parameters.')

def export(master_spec_path, params_path, resource_path, export_path, export_moving_averages):
    if False:
        for i in range(10):
            print('nop')
    'Restores a model and exports it in SavedModel form.\n\n  This method loads a graph specified by the spec at master_spec_path and the\n  params in params_path. It then saves the model in SavedModel format to the\n  location specified in export_path.\n\n  Args:\n    master_spec_path: Path to a proto-text master spec.\n    params_path: Path to the parameters file to export.\n    resource_path: Path to resources in the master spec.\n    export_path: Path to export the SavedModel to.\n    export_moving_averages: Whether to export the moving average parameters.\n  '
    if not tf.gfile.Exists(os.path.join(resource_path, 'known-word-map')):
        with tf.gfile.FastGFile(os.path.join(resource_path, 'known-word-map'), 'w') as out_file:
            out_file.write('This file intentionally left blank.')
    graph = tf.Graph()
    master_spec = spec_pb2.MasterSpec()
    with tf.gfile.FastGFile(master_spec_path) as fin:
        text_format.Parse(fin.read(), master_spec)
    for component in master_spec.component:
        del component.resource[:]
    spec_builder.complete_master_spec(master_spec, None, resource_path)
    stripped_path = export_path.rstrip('/')
    saver_lib.clean_output_paths(stripped_path)
    short_to_original = saver_lib.shorten_resource_paths(master_spec)
    saver_lib.export_master_spec(master_spec, graph)
    saver_lib.export_to_graph(master_spec, params_path, stripped_path, graph, export_moving_averages)
    saver_lib.export_assets(master_spec, short_to_original, stripped_path)

def main(unused_argv):
    if False:
        i = 10
        return i + 15
    export(FLAGS.master_spec, FLAGS.params_path, FLAGS.resource_path, FLAGS.export_path, FLAGS.export_moving_averages)
    tf.logging.info('Export complete.')
if __name__ == '__main__':
    tf.app.run()
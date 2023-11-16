"""Python binary for exporting SavedModel, tailored for TPU inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from object_detection.tpu_exporters import export_saved_model_tpu_lib
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('pipeline_config_file', None, 'A pipeline_pb2.TrainEvalPipelineConfig config file.')
flags.DEFINE_string('ckpt_path', None, 'Path to trained checkpoint, typically of the form path/to/model.ckpt')
flags.DEFINE_string('export_dir', None, 'Path to export SavedModel.')
flags.DEFINE_string('input_placeholder_name', 'placeholder_tensor', "Name of input placeholder in model's signature_def_map.")
flags.DEFINE_string('input_type', 'tf_example', 'Type of input node. Can be one of [`image_tensor`, `encoded_image_string_tensor`, `tf_example`]')
flags.DEFINE_boolean('use_bfloat16', False, 'If true, use tf.bfloat16 on TPU.')

def main(argv):
    if False:
        while True:
            i = 10
    if len(argv) > 1:
        raise tf.app.UsageError('Too many command-line arguments.')
    export_saved_model_tpu_lib.export(FLAGS.pipeline_config_file, FLAGS.ckpt_path, FLAGS.export_dir, FLAGS.input_placeholder_name, FLAGS.input_type, FLAGS.use_bfloat16)
if __name__ == '__main__':
    tf.app.flags.mark_flag_as_required('pipeline_config_file')
    tf.app.flags.mark_flag_as_required('ckpt_path')
    tf.app.flags.mark_flag_as_required('export_dir')
    tf.app.run()
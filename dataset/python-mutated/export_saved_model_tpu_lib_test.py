"""Test for object detection's TPU exporter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from object_detection.tpu_exporters import export_saved_model_tpu_lib
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_path(path_suffix):
    if False:
        print('Hello World!')
    return os.path.join(tf.resource_loader.get_data_files_path(), 'testdata', path_suffix)

class ExportSavedModelTPUTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(('ssd', get_path('ssd/ssd_pipeline.config'), 'image_tensor', True, 20), ('faster_rcnn', get_path('faster_rcnn/faster_rcnn_resnet101_atrous_coco.config'), 'image_tensor', True, 20))
    def testExportAndLoad(self, pipeline_config_file, input_type='image_tensor', use_bfloat16=False, repeat=1):
        if False:
            i = 10
            return i + 15
        input_placeholder_name = 'placeholder_tensor'
        export_dir = os.path.join(FLAGS.test_tmpdir, 'tpu_saved_model')
        if tf.gfile.Exists(export_dir):
            tf.gfile.DeleteRecursively(export_dir)
        ckpt_path = None
        export_saved_model_tpu_lib.export(pipeline_config_file, ckpt_path, export_dir, input_placeholder_name, input_type, use_bfloat16)
        inputs = np.random.rand(256, 256, 3)
        tensor_dict_out = export_saved_model_tpu_lib.run_inference_from_saved_model(inputs, export_dir, input_placeholder_name, repeat)
        for (k, v) in tensor_dict_out.items():
            tf.logging.info('{}: {}'.format(k, v))
if __name__ == '__main__':
    tf.test.main()
"""Tests for export_inference_graph."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import export_inference_graph

class ExportInferenceGraphTest(tf.test.TestCase):

    def testExportInferenceGraph(self):
        if False:
            print('Hello World!')
        tmpdir = self.get_temp_dir()
        output_file = os.path.join(tmpdir, 'inception_v3.pb')
        flags = tf.app.flags.FLAGS
        flags.output_file = output_file
        flags.model_name = 'inception_v3'
        flags.dataset_dir = tmpdir
        export_inference_graph.main(None)
        self.assertTrue(gfile.Exists(output_file))
if __name__ == '__main__':
    tf.test.main()
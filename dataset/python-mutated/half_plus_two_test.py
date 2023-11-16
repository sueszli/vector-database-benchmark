"""Tests for half plus two TF-Hub example."""
import logging
import os
import subprocess
from distutils.version import LooseVersion
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow_hub import test_utils
EXPORT_TOOL_PATH = 'org_tensorflow_hub/examples/half_plus_two/export'

class HalfPlusTwoTest(tf.test.TestCase):

    def testExportTool(self):
        if False:
            for i in range(10):
                print('nop')
        module_path = os.path.join(self.get_temp_dir(), 'half-plus-two-module')
        export_tool_path = os.path.join(test_utils.test_srcdir(), EXPORT_TOOL_PATH)
        self.assertEquals(0, subprocess.call([export_tool_path, module_path]))
        with tf.Graph().as_default():
            m = hub.Module(module_path)
            output = m([10, 3, 4])
            with tf.Session() as session:
                session.run(tf.initializers.global_variables())
                self.assertAllEqual(session.run(output), [7, 3.5, 4])
if __name__ == '__main__':
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        logging.info('Using TF version: %s', tf.__version__)
        tf.test.main()
    else:
        logging.warning('Skipping running tests for TF Version: %s', tf.__version__)
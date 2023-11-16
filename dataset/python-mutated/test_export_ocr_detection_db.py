import os
import shutil
import tempfile
import unittest
from collections import OrderedDict
from modelscope.exporters import Exporter
from modelscope.models import Model
from modelscope.utils.constant import Tasks
from modelscope.utils.test_utils import test_level

class TestExportOCRDetectionDB(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        print('Testing %s.%s' % (type(self).__name__, self._testMethodName))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model_id = 'damo/cv_resnet18_ocr-detection-db-line-level_damo'

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_export_ocr_detection_db(self):
        if False:
            print('Hello World!')
        model = Model.from_pretrained(self.model_id)
        Exporter.from_model(model).export_onnx(input_shape=(1, 3, 800, 800), output_dir=self.tmp_dir)
if __name__ == '__main__':
    unittest.main()
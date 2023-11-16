"""TensorFlow Lite Python Interface: Sanity check."""
import os
import re
from tensorflow.lite.tools import test_utils
from tensorflow.lite.tools import visualize
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class VisualizeTest(test_util.TensorFlowTestCase):

    def testTensorTypeToName(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('FLOAT32', visualize.TensorTypeToName(0))

    def testBuiltinCodeToName(self):
        if False:
            while True:
                i = 10
        self.assertEqual('HASHTABLE_LOOKUP', visualize.BuiltinCodeToName(10))

    def testFlatbufferToDict(self):
        if False:
            return 10
        model = test_utils.build_mock_flatbuffer_model()
        model_dict = visualize.CreateDictFromFlatbuffer(model)
        self.assertEqual(test_utils.TFLITE_SCHEMA_VERSION, model_dict['version'])
        self.assertEqual(1, len(model_dict['subgraphs']))
        self.assertEqual(2, len(model_dict['operator_codes']))
        self.assertEqual(3, len(model_dict['buffers']))
        self.assertEqual(3, len(model_dict['subgraphs'][0]['tensors']))
        self.assertEqual(0, model_dict['subgraphs'][0]['tensors'][0]['buffer'])

    def testVisualize(self):
        if False:
            while True:
                i = 10
        model = test_utils.build_mock_flatbuffer_model()
        tmp_dir = self.get_temp_dir()
        model_filename = os.path.join(tmp_dir, 'model.tflite')
        with open(model_filename, 'wb') as model_file:
            model_file.write(model)
        html_text = visualize.create_html(model_filename)
        self.assertRegex(html_text, re.compile('%s' % model_filename, re.MULTILINE | re.DOTALL))
        self.assertRegex(html_text, re.compile('input_tensor', re.MULTILINE | re.DOTALL))
        self.assertRegex(html_text, re.compile('constant_tensor', re.MULTILINE | re.DOTALL))
        self.assertRegex(html_text, re.compile('ADD', re.MULTILINE | re.DOTALL))
if __name__ == '__main__':
    test.main()
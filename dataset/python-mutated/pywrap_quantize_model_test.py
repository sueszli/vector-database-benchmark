"""Test cases for pywrap_quantize_model.

These test cases are mostly for validation checks. Tests for functionalities
are at `quantize_model_test.py`.
"""
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_quantize_model
from tensorflow.python.platform import test

class PywrapQuantizeModelTest(test.TestCase):
    """Test cases for quantize_model python wrappers."""

    def test_quantize_model_fails_when_invalid_quant_options_serialization(self):
        if False:
            return 10
        saved_model_path = self.create_tempdir('saved_model').full_path
        signature_def_keys = ['serving_default']
        tags = {'serve'}
        quant_opts_serialized = 'invalid protobuf serialization string'
        with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
            pywrap_quantize_model.quantize_ptq_model_pre_calibration(saved_model_path, signature_def_keys, tags, quant_opts_serialized)

    def test_quantize_model_fails_when_invalid_quant_options_type(self):
        if False:
            return 10
        saved_model_path = self.create_tempdir('saved_model').full_path
        signature_def_keys = ['serving_default']
        tags = {'serve'}
        invalid_quant_opts_object = ('a', 'b', 'c')
        with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
            pywrap_quantize_model.quantize_ptq_model_pre_calibration(saved_model_path, signature_def_keys, tags, invalid_quant_opts_object)
if __name__ == '__main__':
    test.main()
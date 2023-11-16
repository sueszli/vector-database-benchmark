"""Target aware conversion for TFLite model."""
from tensorflow.compiler.mlir.lite.experimental.tac.py_wrapper import _pywrap_tac_wrapper

def run_tac(model_path, targets, output_path):
    if False:
        for i in range(10):
            print('nop')
    "Run target aware conversion for the given tflite model file.\n\n  Args:\n    model_path: Path to the tflite model file.\n    targets: A list of string of the desired targets. E.g., ['GPU', 'CPU'].\n    output_path: The output path.\n\n  Returns:\n    Whether the optimization succeeded.\n\n  Raises:\n    ValueError:\n      Invalid model_path.\n      Targets are not specified.\n      Invalid output_path.\n  "
    if not model_path:
        raise ValueError('Invalid model_path.')
    if not targets:
        raise ValueError('Targets are not specified.')
    if not output_path:
        raise ValueError('Invalid output_path.')
    return _pywrap_tac_wrapper.run_tac(model_path, targets, output_path)
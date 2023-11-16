"""SignatureDef utility functions implementation."""
from tensorflow.python.keras.saving.utils_v1 import unexported_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils_impl as utils

def supervised_train_signature_def(inputs, loss, predictions=None, metrics=None):
    if False:
        print('Hello World!')
    return _supervised_signature_def(unexported_constants.SUPERVISED_TRAIN_METHOD_NAME, inputs, loss=loss, predictions=predictions, metrics=metrics)

def supervised_eval_signature_def(inputs, loss, predictions=None, metrics=None):
    if False:
        return 10
    return _supervised_signature_def(unexported_constants.SUPERVISED_EVAL_METHOD_NAME, inputs, loss=loss, predictions=predictions, metrics=metrics)

def _supervised_signature_def(method_name, inputs, loss=None, predictions=None, metrics=None):
    if False:
        return 10
    'Creates a signature for training and eval data.\n\n  This function produces signatures that describe the inputs and outputs\n  of a supervised process, such as training or evaluation, that\n  results in loss, metrics, and the like. Note that this function only requires\n  inputs to be not None.\n\n  Args:\n    method_name: Method name of the SignatureDef as a string.\n    inputs: dict of string to `Tensor`.\n    loss: dict of string to `Tensor` representing computed loss.\n    predictions: dict of string to `Tensor` representing the output predictions.\n    metrics: dict of string to `Tensor` representing metric ops.\n\n  Returns:\n    A train- or eval-flavored signature_def.\n\n  Raises:\n    ValueError: If inputs or outputs is `None`.\n  '
    if inputs is None or not inputs:
        raise ValueError('{} inputs cannot be None or empty.'.format(method_name))
    signature_inputs = {key: utils.build_tensor_info(tensor) for (key, tensor) in inputs.items()}
    signature_outputs = {}
    for output_set in (loss, predictions, metrics):
        if output_set is not None:
            sig_out = {key: utils.build_tensor_info(tensor) for (key, tensor) in output_set.items()}
            signature_outputs.update(sig_out)
    signature_def = signature_def_utils.build_signature_def(signature_inputs, signature_outputs, method_name)
    return signature_def
from bigdl.nano.utils.common import invalidInputError

def PytorchONNXRuntimeModel(model, input_sample=None, onnxruntime_session_options=None, simplification=True, dynamic_axes=True, output_tensors=True, **export_kwargs):
    if False:
        i = 10
        return i + 15
    "\n        Create a ONNX Runtime model from pytorch.\n\n        :param model: 1. Pytorch model to be converted to ONNXRuntime for inference.\n                      2. Path to ONNXRuntime saved model.\n        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or\n                             model is a LightningModule with any dataloader attached,\n                             defaults to None.\n        :param onnxruntime_session_options: A session option for onnxruntime accelerator.\n        :param simplification: whether we use onnxsim to simplify the ONNX model, only valid when\n                               accelerator='onnxruntime', otherwise will be ignored. If this option\n                               is set to True, new dependency 'onnxsim' need to be installed.\n        :param dynamic_axes: dict or boolean, default to True. By default the exported onnx model\n                             will have the first dim of each Tensor input as a dynamic batch_size.\n                             If dynamic_axes=False, the exported model will have the shapes of all\n                             input and output tensors set to exactly match those given in\n                             input_sample. To specify axes of tensors as dynamic (i.e. known only\n                             at run-time), set dynamic_axes to a dict with schema:\n\n                             | KEY (str): an input or output name. Each name must also be provided\n                             | in input_names or output_names.\n                             |\n                             | VALUE (dict or list): If a dict, keys are axis indices and values\n                             | are axis names. If a list, each element is an axis index.\n\n                             If accelerator != 'openvino'/'onnxruntime', it will be ignored.\n        :param output_tensors: boolean, default to True and output of the model will be Tensors.\n                               If output_tensors=False, output of the ONNX model will be ndarray.\n        :param **export_kwargs: will be passed to torch.onnx.export function.\n        :return: A PytorchONNXRuntimeModel instance\n        "
    from .pytorch.pytorch_onnxruntime_model import PytorchONNXRuntimeModel
    return PytorchONNXRuntimeModel(model, input_sample, onnxruntime_session_options=onnxruntime_session_options, simplification=simplification, dynamic_axes=dynamic_axes, output_tensors=output_tensors, **export_kwargs)

def load_onnxruntime_model(path, framework='pytorch'):
    if False:
        print('Hello World!')
    if framework == 'pytorch':
        from .pytorch.pytorch_onnxruntime_model import PytorchONNXRuntimeModel
        return PytorchONNXRuntimeModel._load(path)
    elif framework == 'tensorflow':
        from .tensorflow.model import KerasONNXRuntimeModel
        return KerasONNXRuntimeModel._load(path)
    else:
        invalidInputError(False, "The value {} for framework is not supported. Please choose from 'pytorch'/'tensorflow'.")

def KerasONNXRuntimeModel(model, input_spec, onnxruntime_session_options=None, **export_kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Create a ONNX Runtime model from tensorflow.\n\n    :param model: 1. Keras model to be converted to ONNXRuntime for inference\n                  2. Path to ONNXRuntime saved model\n    :param input_spec: A (tuple or list of) tf.TensorSpec or numpy array defining\n                       the shape/dtype of the input\n    :param onnxruntime_session_options: will be passed to tf2onnx.convert.from_keras function\n    '
    from .tensorflow.model import KerasONNXRuntimeModel
    return KerasONNXRuntimeModel(model, input_spec, onnxruntime_session_options=onnxruntime_session_options, **export_kwargs)
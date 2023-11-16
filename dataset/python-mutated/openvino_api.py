from bigdl.nano.utils.common import invalidInputError

def PytorchOpenVINOModel(model, input_sample=None, precision='fp32', thread_num=None, device='CPU', dynamic_axes=True, logging=True, config=None, output_tensors=True, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Create a OpenVINO model from pytorch.\n\n    :param model: Pytorch model to be converted to OpenVINO for inference or\n                  path to Openvino saved model.\n    :param input_sample: A set of inputs for trace, defaults to None if you have trace before or\n                         model is a LightningModule with any dataloader attached, defaults to None.\n    :param precision: Global precision of model, supported type: 'fp32', 'fp16',\n                      defaults to 'fp32'.\n    :param thread_num: a int represents how many threads(cores) is needed for\n                       inference. default: None.\n    :param device: (optional) A string represents the device of the inference. Default to 'CPU'.\n                   'CPU', 'GPU' and 'VPUX' are supported for now.\n    :param dynamic_axes: dict or boolean, default to True. By default the exported onnx model\n                         will have the first dim of each Tensor input as a dynamic batch_size.\n                         If dynamic_axes=False, the exported model will have the shapes of all\n                         input and output tensors set to exactly match those given in\n                         input_sample. To specify axes of tensors as dynamic (i.e. known only\n                         at run-time), set dynamic_axes to a dict with schema:\n\n                        | KEY (str): an input or output name. Each name must also be provided\n                        | in input_names or output_names.\n                        |\n                        | VALUE (dict or list): If a dict, keys are axis indices and values\n                        | are axis names. If a list, each element is an axis index.\n\n                         If accelerator != 'openvino'/'onnxruntime', it will be ignored.\n    :param logging: whether to log detailed information of model conversion. default: True.\n    :param config: The config to be inputted in core.compile_model.\n    :param output_tensors: boolean, default to True and output of the model will be Tensors.\n                           If output_tensors=False, output of the OpenVINO model will be ndarray.\n    :param **kwargs: will be passed to torch.onnx.export function or model optimizer function.\n    :return: PytorchOpenVINOModel model for OpenVINO inference.\n    "
    from .pytorch.model import PytorchOpenVINOModel
    return PytorchOpenVINOModel(model=model, input_sample=input_sample, precision=precision, thread_num=thread_num, device=device, dynamic_axes=dynamic_axes, logging=logging, config=config, output_tensors=output_tensors, **kwargs)

def load_openvino_model(path, framework='pytorch', device=None, cache_dir=None, shapes=None):
    if False:
        return 10
    "\n    Load an OpenVINO model for inference from directory.\n\n    :param path: Path to model to be loaded.\n    :param framework: Only support pytorch and tensorflow now\n    :param device: A string represents the device of the inference.\n    :param cache_dir: A directory for OpenVINO to cache the model. Default to None.\n    :param shapes: input shape. For example, 'input1[1,3,224,224],input2[1,4]',\n               '[1,3,224,224]'. This parameter affect model Parameter shape, can be\n               dynamic. For dynamic dimesions use symbol `?`, `-1` or range `low.. up`.'.\n               Only valid for openvino model, otherwise will be ignored.\n    :return: PytorchOpenVINOModel model for OpenVINO inference.\n    "
    if cache_dir is not None:
        from pathlib import Path
        Path(cache_dir).mkdir(exist_ok=True)
    if framework == 'pytorch':
        from .pytorch.model import PytorchOpenVINOModel
        return PytorchOpenVINOModel._load(path, device=device, cache_dir=cache_dir, shapes=shapes)
    elif framework == 'tensorflow':
        from .tf.model import KerasOpenVINOModel
        return KerasOpenVINOModel._load(path, device=device, cache_dir=cache_dir, shapes=shapes)
    else:
        invalidInputError(False, "The value {} for framework is not supported. Please choose from 'pytorch'/'tensorflow'.")

def KerasOpenVINOModel(model, input_spec=None, precision='fp32', thread_num=None, device='CPU', config=None, logging=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Create a OpenVINO model from Keras.\n\n    :param model: Keras model to be converted to OpenVINO for inference or\n                  path to Openvino saved model.\n    :param input_spec: A (tuple or list of) tf.TensorSpec or numpy array defining\n                       the shape/dtype of the input\n    :param precision: Global precision of model, supported type: 'fp32', 'fp16',\n                      defaults to 'fp32'.\n    :param thread_num: a int represents how many threads(cores) is needed for\n                       inference. default: None.\n    :param device: (optional) A string represents the device of the inference. Default to 'CPU'.\n                   'CPU', 'GPU' and 'VPUX' are supported for now.\n    :param config: The config to be inputted in core.compile_model.\n    :param logging: whether to log detailed information of model conversion. default: True.\n    :param **kwargs: will be passed to model optimizer function.\n    :return: KerasOpenVINOModel model for OpenVINO inference.\n    "
    from .tf.model import KerasOpenVINOModel
    return KerasOpenVINOModel(model=model, input_spec=input_spec, precision=precision, thread_num=thread_num, device=device, config=config, logging=logging, **kwargs)

def OpenVINOModel(model, device='CPU'):
    if False:
        while True:
            i = 10
    from .core.model import OpenVINOModel
    return OpenVINOModel(model, device)
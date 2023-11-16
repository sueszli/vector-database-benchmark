import warnings
from argparse import ArgumentParser
from os import listdir, makedirs
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from packaging.version import Version, parse
from transformers.pipelines import Pipeline, pipeline
from transformers.tokenization_utils import BatchEncoding
from transformers.utils import ModelOutput, is_tf_available, is_torch_available
ORT_QUANTIZE_MINIMUM_VERSION = parse('1.4.0')
SUPPORTED_PIPELINES = ['feature-extraction', 'ner', 'sentiment-analysis', 'fill-mask', 'question-answering', 'text-generation', 'translation_en_to_fr', 'translation_en_to_de', 'translation_en_to_ro']

class OnnxConverterArgumentParser(ArgumentParser):
    """
    Wraps all the script arguments supported to export transformers models to ONNX IR
    """

    def __init__(self):
        if False:
            return 10
        super().__init__('ONNX Converter')
        self.add_argument('--pipeline', type=str, choices=SUPPORTED_PIPELINES, default='feature-extraction')
        self.add_argument('--model', type=str, required=True, help="Model's id or path (ex: bert-base-cased)")
        self.add_argument('--tokenizer', type=str, help="Tokenizer's id or path (ex: bert-base-cased)")
        self.add_argument('--framework', type=str, choices=['pt', 'tf'], help='Framework for loading the model')
        self.add_argument('--opset', type=int, default=11, help='ONNX opset to use')
        self.add_argument('--check-loading', action='store_true', help='Check ONNX is able to load the model')
        self.add_argument('--use-external-format', action='store_true', help='Allow exporting model >= than 2Gb')
        self.add_argument('--quantize', action='store_true', help='Quantize the neural network to be run with int8')
        self.add_argument('output')

def generate_identified_filename(filename: Path, identifier: str) -> Path:
    if False:
        return 10
    '\n    Append a string-identifier at the end (before the extension, if any) to the provided filepath\n\n    Args:\n        filename: pathlib.Path The actual path object we would like to add an identifier suffix\n        identifier: The suffix to add\n\n    Returns: String with concatenated identifier at the end of the filename\n    '
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)

def check_onnxruntime_requirements(minimum_version: Version):
    if False:
        i = 10
        return i + 15
    '\n    Check onnxruntime is installed and if the installed version match is recent enough\n\n    Raises:\n        ImportError: If onnxruntime is not installed or too old version is found\n    '
    try:
        import onnxruntime
        ort_version = parse(onnxruntime.__version__)
        if ort_version < ORT_QUANTIZE_MINIMUM_VERSION:
            raise ImportError(f'We found an older version of onnxruntime ({onnxruntime.__version__}) but we require onnxruntime to be >= {minimum_version} to enable all the conversions options.\nPlease update onnxruntime by running `pip install --upgrade onnxruntime`')
    except ImportError:
        raise ImportError("onnxruntime doesn't seem to be currently installed. Please install the onnxruntime by running `pip install onnxruntime` and relaunch the conversion.")

def ensure_valid_input(model, tokens, input_names):
    if False:
        i = 10
        return i + 15
    '\n    Ensure inputs are presented in the correct order, without any Non\n\n    Args:\n        model: The model used to forward the input data\n        tokens: BatchEncoding holding the input data\n        input_names: The name of the inputs\n\n    Returns: Tuple\n\n    '
    print('Ensuring inputs are in correct order')
    model_args_name = model.forward.__code__.co_varnames
    (model_args, ordered_input_names) = ([], [])
    for arg_name in model_args_name[1:]:
        if arg_name in input_names:
            ordered_input_names.append(arg_name)
            model_args.append(tokens[arg_name])
        else:
            print(f'{arg_name} is not present in the generated input list.')
            break
    print(f'Generated inputs order: {ordered_input_names}')
    return (ordered_input_names, tuple(model_args))

def infer_shapes(nlp: Pipeline, framework: str) -> Tuple[List[str], List[str], Dict, BatchEncoding]:
    if False:
        return 10
    '\n    Attempt to infer the static vs dynamic axes for each input and output tensors for a specific model\n\n    Args:\n        nlp: The pipeline object holding the model to be exported\n        framework: The framework identifier to dispatch to the correct inference scheme (pt/tf)\n\n    Returns:\n\n        - List of the inferred input variable names\n        - List of the inferred output variable names\n        - Dictionary with input/output variables names as key and shape tensor as value\n        - a BatchEncoding reference which was used to infer all the above information\n    '

    def build_shape_dict(name: str, tensor, is_input: bool, seq_len: int):
        if False:
            i = 10
            return i + 15
        if isinstance(tensor, (tuple, list)):
            return [build_shape_dict(name, t, is_input, seq_len) for t in tensor]
        else:
            axes = {[axis for (axis, numel) in enumerate(tensor.shape) if numel == 1][0]: 'batch'}
            if is_input:
                if len(tensor.shape) == 2:
                    axes[1] = 'sequence'
                else:
                    raise ValueError(f'Unable to infer tensor axes ({len(tensor.shape)})')
            else:
                seq_axes = [dim for (dim, shape) in enumerate(tensor.shape) if shape == seq_len]
                axes.update({dim: 'sequence' for dim in seq_axes})
        print(f"Found {('input' if is_input else 'output')} {name} with shape: {axes}")
        return axes
    tokens = nlp.tokenizer('This is a sample output', return_tensors=framework)
    seq_len = tokens.input_ids.shape[-1]
    outputs = nlp.model(**tokens) if framework == 'pt' else nlp.model(tokens)
    if isinstance(outputs, ModelOutput):
        outputs = outputs.to_tuple()
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
    input_vars = list(tokens.keys())
    input_dynamic_axes = {k: build_shape_dict(k, v, True, seq_len) for (k, v) in tokens.items()}
    outputs_flat = []
    for output in outputs:
        if isinstance(output, (tuple, list)):
            outputs_flat.extend(output)
        else:
            outputs_flat.append(output)
    output_names = [f'output_{i}' for i in range(len(outputs_flat))]
    output_dynamic_axes = {k: build_shape_dict(k, v, False, seq_len) for (k, v) in zip(output_names, outputs_flat)}
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    return (input_vars, output_names, dynamic_axes, tokens)

def load_graph_from_args(pipeline_name: str, framework: str, model: str, tokenizer: Optional[str]=None, **models_kwargs) -> Pipeline:
    if False:
        while True:
            i = 10
    '\n    Convert the set of arguments provided through the CLI to an actual pipeline reference (tokenizer + model\n\n    Args:\n        pipeline_name: The kind of pipeline to use (ner, question-answering, etc.)\n        framework: The actual model to convert the pipeline from ("pt" or "tf")\n        model: The model name which will be loaded by the pipeline\n        tokenizer: The tokenizer name which will be loaded by the pipeline, default to the model\'s value\n\n    Returns: Pipeline object\n\n    '
    if tokenizer is None:
        tokenizer = model
    if framework == 'pt' and (not is_torch_available()):
        raise Exception('Cannot convert because PyTorch is not installed. Please install torch first.')
    if framework == 'tf' and (not is_tf_available()):
        raise Exception('Cannot convert because TF is not installed. Please install tensorflow first.')
    print(f'Loading pipeline (model: {model}, tokenizer: {tokenizer})')
    return pipeline(pipeline_name, model=model, tokenizer=tokenizer, framework=framework, model_kwargs=models_kwargs)

def convert_pytorch(nlp: Pipeline, opset: int, output: Path, use_external_format: bool):
    if False:
        while True:
            i = 10
    '\n    Export a PyTorch backed pipeline to ONNX Intermediate Representation (IR\n\n    Args:\n        nlp: The pipeline to be exported\n        opset: The actual version of the ONNX operator set to use\n        output: Path where will be stored the generated ONNX model\n        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB\n\n    Returns:\n\n    '
    if not is_torch_available():
        raise Exception('Cannot convert because PyTorch is not installed. Please install torch first.')
    import torch
    from torch.onnx import export
    from transformers.pytorch_utils import is_torch_less_than_1_11
    print(f'Using framework PyTorch: {torch.__version__}')
    with torch.no_grad():
        (input_names, output_names, dynamic_axes, tokens) = infer_shapes(nlp, 'pt')
        (ordered_input_names, model_args) = ensure_valid_input(nlp.model, tokens, input_names)
        if is_torch_less_than_1_11:
            export(nlp.model, model_args, f=output.as_posix(), input_names=ordered_input_names, output_names=output_names, dynamic_axes=dynamic_axes, do_constant_folding=True, use_external_data_format=use_external_format, enable_onnx_checker=True, opset_version=opset)
        else:
            export(nlp.model, model_args, f=output.as_posix(), input_names=ordered_input_names, output_names=output_names, dynamic_axes=dynamic_axes, do_constant_folding=True, opset_version=opset)

def convert_tensorflow(nlp: Pipeline, opset: int, output: Path):
    if False:
        print('Hello World!')
    '\n    Export a TensorFlow backed pipeline to ONNX Intermediate Representation (IR)\n\n    Args:\n        nlp: The pipeline to be exported\n        opset: The actual version of the ONNX operator set to use\n        output: Path where will be stored the generated ONNX model\n\n    Notes: TensorFlow cannot export model bigger than 2GB due to internal constraint from TensorFlow\n\n    '
    if not is_tf_available():
        raise Exception('Cannot convert because TF is not installed. Please install tensorflow first.')
    print("/!\\ Please note TensorFlow doesn't support exporting model > 2Gb /!\\")
    try:
        import tensorflow as tf
        import tf2onnx
        from tf2onnx import __version__ as t2ov
        print(f'Using framework TensorFlow: {tf.version.VERSION}, tf2onnx: {t2ov}')
        (input_names, output_names, dynamic_axes, tokens) = infer_shapes(nlp, 'tf')
        nlp.model.predict(tokens.data)
        input_signature = [tf.TensorSpec.from_tensor(tensor, name=key) for (key, tensor) in tokens.items()]
        (model_proto, _) = tf2onnx.convert.from_keras(nlp.model, input_signature, opset=opset, output_path=output.as_posix())
    except ImportError as e:
        raise Exception(f'Cannot import {e.name} required to convert TF model to ONNX. Please install {e.name} first. {e}')

def convert(framework: str, model: str, output: Path, opset: int, tokenizer: Optional[str]=None, use_external_format: bool=False, pipeline_name: str='feature-extraction', **model_kwargs):
    if False:
        while True:
            i = 10
    '\n    Convert the pipeline object to the ONNX Intermediate Representation (IR) format\n\n    Args:\n        framework: The framework the pipeline is backed by ("pt" or "tf")\n        model: The name of the model to load for the pipeline\n        output: The path where the ONNX graph will be stored\n        opset: The actual version of the ONNX operator set to use\n        tokenizer: The name of the model to load for the pipeline, default to the model\'s name if not provided\n        use_external_format:\n            Split the model definition from its parameters to allow model bigger than 2GB (PyTorch only)\n        pipeline_name: The kind of pipeline to instantiate (ner, question-answering, etc.)\n        model_kwargs: Keyword arguments to be forwarded to the model constructor\n\n    Returns:\n\n    '
    warnings.warn('The `transformers.convert_graph_to_onnx` package is deprecated and will be removed in version 5 of Transformers', FutureWarning)
    print(f'ONNX opset version set to: {opset}')
    nlp = load_graph_from_args(pipeline_name, framework, model, tokenizer, **model_kwargs)
    if not output.parent.exists():
        print(f'Creating folder {output.parent}')
        makedirs(output.parent.as_posix())
    elif len(listdir(output.parent.as_posix())) > 0:
        raise Exception(f'Folder {output.parent.as_posix()} is not empty, aborting conversion')
    if framework == 'pt':
        convert_pytorch(nlp, opset, output, use_external_format)
    else:
        convert_tensorflow(nlp, opset, output)

def optimize(onnx_model_path: Path) -> Path:
    if False:
        i = 10
        return i + 15
    '\n    Load the model at the specified path and let onnxruntime look at transformations on the graph to enable all the\n    optimizations possible\n\n    Args:\n        onnx_model_path: filepath where the model binary description is stored\n\n    Returns: Path where the optimized model binary description has been saved\n\n    '
    from onnxruntime import InferenceSession, SessionOptions
    opt_model_path = generate_identified_filename(onnx_model_path, '-optimized')
    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    _ = InferenceSession(onnx_model_path.as_posix(), sess_option)
    print(f'Optimized model has been written at {opt_model_path}: ✔')
    print('/!\\ Optimized model contains hardware specific operators which might not be portable. /!\\')
    return opt_model_path

def quantize(onnx_model_path: Path) -> Path:
    if False:
        i = 10
        return i + 15
    '\n    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU\n\n    Args:\n        onnx_model_path: Path to location the exported ONNX model is stored\n\n    Returns: The Path generated for the quantized\n    '
    import onnx
    import onnxruntime
    from onnx.onnx_pb import ModelProto
    from onnxruntime.quantization import QuantizationMode
    from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
    from onnxruntime.quantization.registry import IntegerOpsRegistry
    onnx_model = onnx.load(onnx_model_path.as_posix())
    if parse(onnx.__version__) < parse('1.5.0'):
        print('Models larger than 2GB will fail to quantize due to protobuf constraint.\nPlease upgrade to onnxruntime >= 1.5.0.')
    copy_model = ModelProto()
    copy_model.CopyFrom(onnx_model)
    if parse(onnxruntime.__version__) < parse('1.13.1'):
        quantizer = ONNXQuantizer(model=copy_model, per_channel=False, reduce_range=False, mode=QuantizationMode.IntegerOps, static=False, weight_qType=True, input_qType=False, tensors_range=None, nodes_to_quantize=None, nodes_to_exclude=None, op_types_to_quantize=list(IntegerOpsRegistry))
    else:
        quantizer = ONNXQuantizer(model=copy_model, per_channel=False, reduce_range=False, mode=QuantizationMode.IntegerOps, static=False, weight_qType=True, activation_qType=False, tensors_range=None, nodes_to_quantize=None, nodes_to_exclude=None, op_types_to_quantize=list(IntegerOpsRegistry))
    quantizer.quantize_model()
    quantized_model_path = generate_identified_filename(onnx_model_path, '-quantized')
    print(f'Quantized model has been written at {quantized_model_path}: ✔')
    onnx.save_model(quantizer.model.model, quantized_model_path.as_posix())
    return quantized_model_path

def verify(path: Path):
    if False:
        return 10
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
    print(f'Checking ONNX model loading from: {path} ...')
    try:
        onnx_options = SessionOptions()
        _ = InferenceSession(path.as_posix(), onnx_options, providers=['CPUExecutionProvider'])
        print(f'Model {path} correctly loaded: ✔')
    except RuntimeException as re:
        print(f'Error while loading the model {re}: ✘')
if __name__ == '__main__':
    parser = OnnxConverterArgumentParser()
    args = parser.parse_args()
    args.output = Path(args.output).absolute()
    try:
        print('\n====== Converting model to ONNX ======')
        convert(args.framework, args.model, args.output, args.opset, args.tokenizer, args.use_external_format, args.pipeline)
        if args.quantize:
            check_onnxruntime_requirements(ORT_QUANTIZE_MINIMUM_VERSION)
            if args.framework == 'tf':
                print('\t Using TensorFlow might not provide the same optimization level compared to PyTorch.\n\t For TensorFlow users you can try optimizing the model directly through onnxruntime_tools.\n\t For more information, please refer to the onnxruntime documentation:\n\t\thttps://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers\n')
            print('\n====== Optimizing ONNX model ======')
            args.optimized_output = optimize(args.output)
            args.quantized_output = quantize(args.optimized_output)
        if args.check_loading:
            print('\n====== Check exported ONNX model(s) ======')
            verify(args.output)
            if hasattr(args, 'optimized_output'):
                verify(args.optimized_output)
            if hasattr(args, 'quantized_output'):
                verify(args.quantized_output)
    except Exception as e:
        print(f'Error while converting the model: {e}')
        exit(1)
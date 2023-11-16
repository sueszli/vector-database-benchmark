import logging
import sys
from collections import defaultdict
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Union
import tensorflow as tf
import torch
from apache_beam.ml.inference import utils
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.pytorch_inference import _convert_to_device
from transformers import AutoModel
from transformers import Pipeline
from transformers import TFAutoModel
from transformers import pipeline
_LOGGER = logging.getLogger(__name__)
__all__ = ['HuggingFaceModelHandlerTensor', 'HuggingFaceModelHandlerKeyedTensor', 'HuggingFacePipelineModelHandler']
TensorInferenceFn = Callable[[Sequence[Union[torch.Tensor, tf.Tensor]], Union[AutoModel, TFAutoModel], str, Optional[Dict[str, Any]], Optional[str]], Iterable[PredictionResult]]
KeyedTensorInferenceFn = Callable[[Sequence[Dict[str, Union[torch.Tensor, tf.Tensor]]], Union[AutoModel, TFAutoModel], str, Optional[Dict[str, Any]], Optional[str]], Iterable[PredictionResult]]
PipelineInferenceFn = Callable[[Sequence[str], Pipeline, Optional[Dict[str, Any]]], Iterable[PredictionResult]]

class PipelineTask(str, Enum):
    """
  PipelineTask defines all the tasks supported by the Hugging Face Pipelines
  listed at https://huggingface.co/docs/transformers/main_classes/pipelines.
  Only these tasks can be passed to HuggingFacePipelineModelHandler.
  """
    AudioClassification = 'audio-classification'
    AutomaticSpeechRecognition = 'automatic-speech-recognition'
    Conversational = 'conversational'
    DepthEstimation = 'depth-estimation'
    DocumentQuestionAnswering = 'document-question-answering'
    FeatureExtraction = 'feature-extraction'
    FillMask = 'fill-mask'
    ImageClassification = 'image-classification'
    ImageSegmentation = 'image-segmentation'
    ImageToText = 'image-to-text'
    MaskGeneration = 'mask-generation'
    NER = 'ner'
    ObjectDetection = 'object-detection'
    QuestionAnswering = 'question-answering'
    SentimentAnalysis = 'sentiment-analysis'
    Summarization = 'summarization'
    TableQuestionAnswering = 'table-question-answering'
    TextClassification = 'text-classification'
    TextGeneration = 'text-generation'
    Text2TextGeneration = 'text2text-generation'
    TextToAudio = 'text-to-audio'
    TokenClassification = 'token-classification'
    Translation = 'translation'
    VideoClassification = 'video-classification'
    VisualQuestionAnswering = 'visual-question-answering'
    VQA = 'vqa'
    ZeroShotAudioClassification = 'zero-shot-audio-classification'
    ZeroShotClassification = 'zero-shot-classification'
    ZeroShotImageClassification = 'zero-shot-image-classification'
    ZeroShotObjectDetection = 'zero-shot-object-detection'
    Translation_XX_to_YY = 'translation_XX_to_YY'

def _validate_constructor_args(model_uri, model_class):
    if False:
        while True:
            i = 10
    message = 'Please provide both model class and model uri to load the model.Got params as model_uri={model_uri} and model_class={model_class}.'
    if not model_uri and (not model_class):
        raise RuntimeError(message.format(model_uri=model_uri, model_class=model_class))
    elif not model_uri:
        raise RuntimeError(message.format(model_uri=model_uri, model_class=model_class))
    elif not model_class:
        raise RuntimeError(message.format(model_uri=model_uri, model_class=model_class))

def no_gpu_available_warning():
    if False:
        print('Hello World!')
    _LOGGER.warning("HuggingFaceModelHandler specified a 'GPU' device, but GPUs are not available. Switching to CPU.")

def is_gpu_available_torch():
    if False:
        while True:
            i = 10
    if torch.cuda.is_available():
        return True
    else:
        no_gpu_available_warning()
        return False

def get_device_torch(device):
    if False:
        print('Hello World!')
    if device == 'GPU' and is_gpu_available_torch():
        return torch.device('cuda')
    return torch.device('cpu')

def is_gpu_available_tensorflow(device):
    if False:
        i = 10
        return i + 15
    gpu_devices = tf.config.list_physical_devices(device)
    if len(gpu_devices) == 0:
        no_gpu_available_warning()
        return False
    return True

def _validate_constructor_args_hf_pipeline(task, model):
    if False:
        while True:
            i = 10
    if not task and (not model):
        raise RuntimeError('Please provide either task or model to the HuggingFacePipelineModelHandler. If the model already defines the task, no need to specify the task.')

def _run_inference_torch_keyed_tensor(batch: Sequence[Dict[str, torch.Tensor]], model: AutoModel, device, inference_args: Dict[str, Any], model_id: Optional[str]=None) -> Iterable[PredictionResult]:
    if False:
        for i in range(10):
            print('nop')
    device = get_device_torch(device)
    key_to_tensor_list = defaultdict(list)
    with torch.no_grad():
        for example in batch:
            for (key, tensor) in example.items():
                key_to_tensor_list[key].append(tensor)
        key_to_batched_tensors = {}
        for key in key_to_tensor_list:
            batched_tensors = torch.stack(key_to_tensor_list[key])
            batched_tensors = _convert_to_device(batched_tensors, device)
            key_to_batched_tensors[key] = batched_tensors
        predictions = model(**key_to_batched_tensors, **inference_args)
        return utils._convert_to_result(batch, predictions, model_id)

def _run_inference_tensorflow_keyed_tensor(batch: Sequence[Dict[str, tf.Tensor]], model: TFAutoModel, device, inference_args: Dict[str, Any], model_id: Optional[str]=None) -> Iterable[PredictionResult]:
    if False:
        for i in range(10):
            print('nop')
    if device == 'GPU':
        is_gpu_available_tensorflow(device)
    key_to_tensor_list = defaultdict(list)
    for example in batch:
        for (key, tensor) in example.items():
            key_to_tensor_list[key].append(tensor)
    key_to_batched_tensors = {}
    for key in key_to_tensor_list:
        batched_tensors = tf.stack(key_to_tensor_list[key], axis=0)
        key_to_batched_tensors[key] = batched_tensors
    predictions = model(**key_to_batched_tensors, **inference_args)
    return utils._convert_to_result(batch, predictions, model_id)

class HuggingFaceModelHandlerKeyedTensor(ModelHandler[Dict[str, Union[tf.Tensor, torch.Tensor]], PredictionResult, Union[AutoModel, TFAutoModel]]):

    def __init__(self, model_uri: str, model_class: Union[AutoModel, TFAutoModel], framework: str, device: str='CPU', *, inference_fn: Optional[Callable[..., Iterable[PredictionResult]]]=None, load_model_args: Optional[Dict[str, Any]]=None, inference_args: Optional[Dict[str, Any]]=None, min_batch_size: Optional[int]=None, max_batch_size: Optional[int]=None, large_model: bool=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n    Implementation of the ModelHandler interface for HuggingFace with\n    Keyed Tensors for PyTorch/Tensorflow backend.\n\n    Example Usage model::\n      pcoll | RunInference(HuggingFaceModelHandlerKeyedTensor(\n        model_uri="bert-base-uncased", model_class=AutoModelForMaskedLM,\n        framework=\'pt\'))\n\n    Args:\n      model_uri (str): path to the pretrained model on the hugging face\n        models hub.\n      model_class: model class to load the repository from model_uri.\n      framework (str): Framework to use for the model. \'tf\' for TensorFlow and\n        \'pt\' for PyTorch.\n      device: For torch tensors, specify device on which you wish to\n        run the model. Defaults to CPU.\n      inference_fn: the inference function to use during RunInference.\n        Default is _run_inference_torch_keyed_tensor or\n        _run_inference_tensorflow_keyed_tensor depending on the input type.\n      load_model_args (Dict[str, Any]): (Optional) Keyword arguments to provide\n        load options while loading models from Hugging Face Hub.\n        Defaults to None.\n      inference_args (Dict[str, Any]): (Optional) Non-batchable arguments\n        required as inputs to the model\'s inference function. Unlike Tensors\n        in `batch`, these parameters will not be dynamically batched.\n        Defaults to None.\n      min_batch_size: the minimum batch size to use when batching inputs.\n      max_batch_size: the maximum batch size to use when batching inputs.\n      large_model: set to true if your model is large enough to run into\n        memory pressure if you load multiple copies. Given a model that\n        consumes N memory and a machine with W cores and M memory, you should\n        set this to True if N*W > M.\n      kwargs: \'env_vars\' can be used to set environment variables\n        before loading the model.\n\n    **Supported Versions:** HuggingFaceModelHandler supports\n    transformers>=4.18.0.\n    '
        self._model_uri = model_uri
        self._model_class = model_class
        self._device = device
        self._inference_fn = inference_fn
        self._model_config_args = load_model_args if load_model_args else {}
        self._inference_args = inference_args if inference_args else {}
        self._batching_kwargs = {}
        self._env_vars = kwargs.get('env_vars', {})
        if min_batch_size is not None:
            self._batching_kwargs['min_batch_size'] = min_batch_size
        if max_batch_size is not None:
            self._batching_kwargs['max_batch_size'] = max_batch_size
        self._large_model = large_model
        self._framework = framework
        _validate_constructor_args(model_uri=self._model_uri, model_class=self._model_class)

    def load_model(self):
        if False:
            print('Hello World!')
        'Loads and initializes the model for processing.'
        model = self._model_class.from_pretrained(self._model_uri, **self._model_config_args)
        if self._framework == 'pt':
            if self._device == 'GPU' and is_gpu_available_torch:
                model.to(torch.device('cuda'))
            if callable(getattr(model, 'requires_grad_', None)):
                model.requires_grad_(False)
        return model

    def run_inference(self, batch: Sequence[Dict[str, Union[tf.Tensor, torch.Tensor]]], model: Union[AutoModel, TFAutoModel], inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            for i in range(10):
                print('nop')
        "\n    Runs inferences on a batch of Keyed Tensors and returns an Iterable of\n    Tensors Predictions.\n\n    This method stacks the list of Tensors in a vectorized format to optimize\n    the inference call.\n\n    Args:\n      batch: A sequence of Keyed Tensors. These Tensors should be batchable,\n        as this method will call `tf.stack()`/`torch.stack()` and pass in\n        batched Tensors with dimensions (batch_size, n_features, etc.) into\n        the model's predict() function.\n      model: A Tensorflow/PyTorch model.\n      inference_args: Non-batchable arguments required as inputs to the\n        model's inference function. Unlike Tensors in `batch`,\n        these parameters will not be dynamically batched.\n    Returns:\n      An Iterable of type PredictionResult.\n    "
        inference_args = {} if not inference_args else inference_args
        if self._inference_fn:
            return self._inference_fn(batch, model, self._device, inference_args, self._model_uri)
        if self._framework == 'tf':
            return _run_inference_tensorflow_keyed_tensor(batch, model, self._device, inference_args, self._model_uri)
        else:
            return _run_inference_torch_keyed_tensor(batch, model, self._device, inference_args, self._model_uri)

    def update_model_path(self, model_path: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        self._model_uri = model_path if model_path else self._model_uri

    def get_num_bytes(self, batch: Sequence[Union[tf.Tensor, torch.Tensor]]) -> int:
        if False:
            i = 10
            return i + 15
        '\n    Returns:\n      The number of bytes of data for the Tensors batch.\n    '
        if self._framework == 'tf':
            return sum((sys.getsizeof(element) for element in batch))
        else:
            return sum((el.element_size() for tensor in batch for el in tensor.values()))

    def batch_elements_kwargs(self):
        if False:
            while True:
                i = 10
        return self._batching_kwargs

    def share_model_across_processes(self) -> bool:
        if False:
            return 10
        return self._large_model

    def get_metrics_namespace(self) -> str:
        if False:
            print('Hello World!')
        '\n    Returns:\n        A namespace for metrics collected by the RunInference transform.\n    '
        return 'BeamML_HuggingFaceModelHandler_KeyedTensor'

def _default_inference_fn_torch(batch: Sequence[Union[tf.Tensor, torch.Tensor]], model: Union[AutoModel, TFAutoModel], device, inference_args: Dict[str, Any], model_id: Optional[str]=None) -> Iterable[PredictionResult]:
    if False:
        while True:
            i = 10
    device = get_device_torch(device)
    with torch.no_grad():
        batched_tensors = torch.stack(batch)
        batched_tensors = _convert_to_device(batched_tensors, device)
        predictions = model(batched_tensors, **inference_args)
    return utils._convert_to_result(batch, predictions, model_id)

def _default_inference_fn_tensorflow(batch: Sequence[Union[tf.Tensor, torch.Tensor]], model: Union[AutoModel, TFAutoModel], device, inference_args: Dict[str, Any], model_id: Optional[str]=None) -> Iterable[PredictionResult]:
    if False:
        i = 10
        return i + 15
    if device == 'GPU':
        is_gpu_available_tensorflow(device)
    batched_tensors = tf.stack(batch, axis=0)
    predictions = model(batched_tensors, **inference_args)
    return utils._convert_to_result(batch, predictions, model_id)

class HuggingFaceModelHandlerTensor(ModelHandler[Union[tf.Tensor, torch.Tensor], PredictionResult, Union[AutoModel, TFAutoModel]]):

    def __init__(self, model_uri: str, model_class: Union[AutoModel, TFAutoModel], device: str='CPU', *, inference_fn: Optional[Callable[..., Iterable[PredictionResult]]]=None, load_model_args: Optional[Dict[str, Any]]=None, inference_args: Optional[Dict[str, Any]]=None, min_batch_size: Optional[int]=None, max_batch_size: Optional[int]=None, large_model: bool=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n    Implementation of the ModelHandler interface for HuggingFace with\n    Tensors for PyTorch/Tensorflow backend.\n\n    Depending on the type of tensors, the model framework is determined\n    automatically.\n\n    Example Usage model:\n      pcoll | RunInference(HuggingFaceModelHandlerTensor(\n        model_uri="bert-base-uncased", model_class=AutoModelForMaskedLM))\n\n    Args:\n      model_uri (str): path to the pretrained model on the hugging face\n        models hub.\n      model_class: model class to load the repository from model_uri.\n      device: For torch tensors, specify device on which you wish to\n        run the model. Defaults to CPU.\n      inference_fn: the inference function to use during RunInference.\n        Default is _run_inference_torch_keyed_tensor or\n        _run_inference_tensorflow_keyed_tensor depending on the input type.\n      load_model_args (Dict[str, Any]): (Optional) keyword arguments to provide\n        load options while loading models from Hugging Face Hub.\n        Defaults to None.\n      inference_args (Dict[str, Any]): (Optional) Non-batchable arguments\n        required as inputs to the model\'s inference function. Unlike Tensors\n        in `batch`, these parameters will not be dynamically batched.\n        Defaults to None.\n      min_batch_size: the minimum batch size to use when batching inputs.\n      max_batch_size: the maximum batch size to use when batching inputs.\n      large_model: set to true if your model is large enough to run into\n        memory pressure if you load multiple copies. Given a model that\n        consumes N memory and a machine with W cores and M memory, you should\n        set this to True if N*W > M.\n      kwargs: \'env_vars\' can be used to set environment variables\n        before loading the model.\n\n    **Supported Versions:** HuggingFaceModelHandler supports\n    transformers>=4.18.0.\n    '
        self._model_uri = model_uri
        self._model_class = model_class
        self._device = device
        self._inference_fn = inference_fn
        self._model_config_args = load_model_args if load_model_args else {}
        self._inference_args = inference_args if inference_args else {}
        self._batching_kwargs = {}
        self._env_vars = kwargs.get('env_vars', {})
        if min_batch_size is not None:
            self._batching_kwargs['min_batch_size'] = min_batch_size
        if max_batch_size is not None:
            self._batching_kwargs['max_batch_size'] = max_batch_size
        self._large_model = large_model
        self._framework = ''
        _validate_constructor_args(model_uri=self._model_uri, model_class=self._model_class)

    def load_model(self):
        if False:
            for i in range(10):
                print('nop')
        'Loads and initializes the model for processing.'
        model = self._model_class.from_pretrained(self._model_uri, **self._model_config_args)
        if callable(getattr(model, 'requires_grad_', None)):
            model.requires_grad_(False)
        return model

    def run_inference(self, batch: Sequence[Union[tf.Tensor, torch.Tensor]], model: Union[AutoModel, TFAutoModel], inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            while True:
                i = 10
        "\n    Runs inferences on a batch of Tensors and returns an Iterable of\n    Tensors Predictions.\n\n    This method stacks the list of Tensors in a vectorized format to optimize\n    the inference call.\n\n    Args:\n      batch: A sequence of Tensors. These Tensors should be batchable, as\n        this method will call `tf.stack()`/`torch.stack()` and pass in\n        batched Tensors with dimensions (batch_size, n_features, etc.)\n        into the model's predict() function.\n      model: A Tensorflow/PyTorch model.\n      inference_args (Dict[str, Any]): Non-batchable arguments required as\n        inputs to the model's inference function. Unlike Tensors in `batch`,\n        these parameters will not be dynamically batched.\n\n    Returns:\n      An Iterable of type PredictionResult.\n    "
        inference_args = {} if not inference_args else inference_args
        if not self._framework:
            if isinstance(batch[0], tf.Tensor):
                self._framework = 'tf'
            else:
                self._framework = 'pt'
        if self._framework == 'pt' and self._device == 'GPU' and is_gpu_available_torch():
            model.to(torch.device('cuda'))
        if self._inference_fn:
            return self._inference_fn(batch, model, inference_args, inference_args, self._model_uri)
        if self._framework == 'tf':
            return _default_inference_fn_tensorflow(batch, model, self._device, inference_args, self._model_uri)
        else:
            return _default_inference_fn_torch(batch, model, self._device, inference_args, self._model_uri)

    def update_model_path(self, model_path: Optional[str]=None):
        if False:
            print('Hello World!')
        self._model_uri = model_path if model_path else self._model_uri

    def get_num_bytes(self, batch: Sequence[Union[tf.Tensor, torch.Tensor]]) -> int:
        if False:
            print('Hello World!')
        '\n    Returns:\n      The number of bytes of data for the Tensors batch.\n    '
        if self._framework == 'tf':
            return sum((sys.getsizeof(element) for element in batch))
        else:
            return sum((el.element_size() for tensor in batch for el in tensor.values()))

    def batch_elements_kwargs(self):
        if False:
            while True:
                i = 10
        return self._batching_kwargs

    def share_model_across_processes(self) -> bool:
        if False:
            while True:
                i = 10
        return self._large_model

    def get_metrics_namespace(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n    Returns:\n       A namespace for metrics collected by the RunInference transform.\n    '
        return 'BeamML_HuggingFaceModelHandler_Tensor'

def _convert_to_result(batch: Iterable, predictions: Union[Iterable, Dict[Any, Iterable]], model_id: Optional[str]=None) -> Iterable[PredictionResult]:
    if False:
        print('Hello World!')
    return [PredictionResult(x, y, model_id) for (x, y) in zip(batch, [predictions])]

def _default_pipeline_inference_fn(batch, pipeline, inference_args) -> Iterable[PredictionResult]:
    if False:
        while True:
            i = 10
    predicitons = pipeline(batch, **inference_args)
    return predicitons

class HuggingFacePipelineModelHandler(ModelHandler[str, PredictionResult, Pipeline]):

    def __init__(self, task: Union[str, PipelineTask]='', model: str='', *, device: Optional[str]=None, inference_fn: PipelineInferenceFn=_default_pipeline_inference_fn, load_pipeline_args: Optional[Dict[str, Any]]=None, inference_args: Optional[Dict[str, Any]]=None, min_batch_size: Optional[int]=None, max_batch_size: Optional[int]=None, large_model: bool=False, **kwargs):
        if False:
            print('Hello World!')
        '\n    Implementation of the ModelHandler interface for Hugging Face Pipelines.\n\n    Example Usage model::\n      pcoll | RunInference(HuggingFacePipelineModelHandler(\n        task="fill-mask"))\n\n    Args:\n      task (str or enum.Enum): task supported by HuggingFace Pipelines.\n        Accepts a string task or an enum.Enum from PipelineTask.\n      model (str): path to the pretrained *model-id* on Hugging Face Models Hub\n        to use custom model for the chosen task. If the `model` already defines\n        the task then no need to specify the `task` parameter.\n        Use the *model-id* string instead of an actual model here.\n        Model-specific kwargs for `from_pretrained(..., **model_kwargs)` can be\n        specified with `model_kwargs` using `load_pipeline_args`.\n\n        Example Usage::\n          model_handler = HuggingFacePipelineModelHandler(\n            task="text-generation", model="meta-llama/Llama-2-7b-hf",\n            load_pipeline_args={\'model_kwargs\':{\'quantization_map\':config}})\n\n      device (str): the device (`"CPU"` or `"GPU"`) on which you wish to run\n        the pipeline. Defaults to GPU. If GPU is not available then it falls\n        back to CPU. You can also use advanced option like `device_map` with\n        key-value pair as you would do in the usual Hugging Face pipeline using\n        `load_pipeline_args`. Ex: load_pipeline_args={\'device_map\':auto}).\n      inference_fn: the inference function to use during RunInference.\n        Default is _default_pipeline_inference_fn.\n      load_pipeline_args (Dict[str, Any]): keyword arguments to provide load\n        options while loading pipelines from Hugging Face. Defaults to None.\n      inference_args (Dict[str, Any]): Non-batchable arguments\n        required as inputs to the model\'s inference function.\n        Defaults to None.\n      min_batch_size: the minimum batch size to use when batching inputs.\n      max_batch_size: the maximum batch size to use when batching inputs.\n      large_model: set to true if your model is large enough to run into\n        memory pressure if you load multiple copies. Given a model that\n        consumes N memory and a machine with W cores and M memory, you should\n        set this to True if N*W > M.\n      kwargs: \'env_vars\' can be used to set environment variables\n        before loading the model.\n\n    **Supported Versions:** HuggingFacePipelineModelHandler supports\n    transformers>=4.18.0.\n    '
        self._task = task
        self._model = model
        self._inference_fn = inference_fn
        self._load_pipeline_args = load_pipeline_args if load_pipeline_args else {}
        self._inference_args = inference_args if inference_args else {}
        self._batching_kwargs = {}
        self._framework = 'torch'
        self._env_vars = kwargs.get('env_vars', {})
        if min_batch_size is not None:
            self._batching_kwargs['min_batch_size'] = min_batch_size
        if max_batch_size is not None:
            self._batching_kwargs['max_batch_size'] = max_batch_size
        self._large_model = large_model
        self._deduplicate_device_value(device)
        _validate_constructor_args_hf_pipeline(self._task, self._model)

    def _deduplicate_device_value(self, device: Optional[str]):
        if False:
            return 10
        current_device = device.upper() if device else None
        if current_device and current_device != 'CPU' and (current_device != 'GPU'):
            raise ValueError(f'Invalid device value: {device}. Please specify either CPU or GPU. Defaults to GPU if no value is provided.')
        if 'device' not in self._load_pipeline_args:
            if current_device == 'CPU':
                self._load_pipeline_args['device'] = 'cpu'
            elif is_gpu_available_torch():
                self._load_pipeline_args['device'] = 'cuda:1'
            else:
                _LOGGER.warning("HuggingFaceModelHandler specified a 'GPU' device, but GPUs are not available. Switching to CPU.")
                self._load_pipeline_args['device'] = 'cpu'
        elif current_device:
            raise ValueError('`device` specified in `load_pipeline_args`. `device` parameter for HuggingFacePipelineModelHandler will be ignored.')

    def load_model(self):
        if False:
            for i in range(10):
                print('nop')
        'Loads and initializes the pipeline for processing.'
        return pipeline(task=self._task, model=self._model, **self._load_pipeline_args)

    def run_inference(self, batch: Sequence[str], pipeline: Pipeline, inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            print('Hello World!')
        "\n    Runs inferences on a batch of examples passed as a string resource.\n    These can either be string sentences, or string path to images or\n    audio files.\n\n    Args:\n      batch: A sequence of strings resources.\n      pipeline: A Hugging Face Pipeline.\n      inference_args: Non-batchable arguments required as inputs to the model's\n        inference function.\n    Returns:\n      An Iterable of type PredictionResult.\n    "
        inference_args = {} if not inference_args else inference_args
        predictions = self._inference_fn(batch, pipeline, inference_args)
        return _convert_to_result(batch, predictions)

    def update_model_path(self, model_path: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        '\n    Updates the pretrained model used by the Hugging Face Pipeline task.\n    Make sure that the new model does the same task as initial model.\n\n    Args:\n      model_path (str): (Optional) Path to the new trained model\n        from Hugging Face. Defaults to None.\n    '
        self._model = model_path if model_path else self._model

    def get_num_bytes(self, batch: Sequence[str]) -> int:
        if False:
            return 10
        '\n    Returns:\n      The number of bytes of input batch elements.\n    '
        return sum((sys.getsizeof(element) for element in batch))

    def batch_elements_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        return self._batching_kwargs

    def share_model_across_processes(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._large_model

    def get_metrics_namespace(self) -> str:
        if False:
            while True:
                i = 10
        '\n    Returns:\n       A namespace for metrics collected by the RunInference transform.\n    '
        return 'BeamML_HuggingFacePipelineModelHandler'
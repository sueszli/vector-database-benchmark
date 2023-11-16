from __future__ import annotations
import logging
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple
import numpy as np
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference import utils
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult
LOGGER = logging.getLogger('TensorRTEngineHandlerNumPy')
try:
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace='')
    LOGGER.info('tensorrt module successfully imported.')
except ModuleNotFoundError:
    TRT_LOGGER = None
    msg = 'tensorrt module was not found. This is ok as long as the specified runner has tensorrt dependencies installed.'
    LOGGER.warning(msg)

def _load_engine(engine_path):
    if False:
        i = 10
        return i + 15
    import tensorrt as trt
    file = FileSystems.open(engine_path, 'rb')
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(file.read())
    assert engine
    return engine

def _load_onnx(onnx_path):
    if False:
        print('Hello World!')
    import tensorrt as trt
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with FileSystems.open(onnx_path) as f:
        if not parser.parse(f.read()):
            LOGGER.error('Failed to load ONNX file: %s', onnx_path)
            for error in range(parser.num_errors):
                LOGGER.error(parser.get_error(error))
            raise ValueError(f'Failed to load ONNX file: {onnx_path}')
    return (network, builder)

def _build_engine(network, builder):
    if False:
        i = 10
        return i + 15
    import tensorrt as trt
    config = builder.create_builder_config()
    runtime = trt.Runtime(TRT_LOGGER)
    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)
    builder.reset()
    return engine

def _assign_or_fail(args):
    if False:
        i = 10
        return i + 15
    'CUDA error checking.'
    from cuda import cuda
    (err, ret) = (args[0], args[1:])
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Cuda Error: {}'.format(err))
    else:
        raise RuntimeError('Unknown error type: {}'.format(err))
    if len(ret) == 1:
        return ret[0]
    return ret

class TensorRTEngine:

    def __init__(self, engine: trt.ICudaEngine):
        if False:
            for i in range(10):
                print('nop')
        'Implementation of the TensorRTEngine class which handles\n    allocations associated with TensorRT engine.\n\n    Example Usage::\n\n      TensorRTEngine(engine)\n\n    Args:\n      engine: trt.ICudaEngine object that contains TensorRT engine\n    '
        from cuda import cuda
        import tensorrt as trt
        self.engine = engine
        self.context = engine.create_execution_context()
        self.context_lock = threading.RLock()
        self.inputs = []
        self.outputs = []
        self.gpu_allocations = []
        self.cpu_allocations = []
        try:
            _ = np.bool
        except AttributeError:
            np.bool = np.bool_
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape) * dtype.itemsize
            allocation = _assign_or_fail(cuda.cuMemAlloc(size))
            binding = {'index': i, 'name': name, 'dtype': np.dtype(trt.nptype(dtype)), 'shape': list(shape), 'allocation': allocation, 'size': size}
            self.gpu_allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        assert self.context
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.gpu_allocations) > 0
        for output in self.outputs:
            self.cpu_allocations.append(np.zeros(output['shape'], output['dtype']))
        self.stream = _assign_or_fail(cuda.cuStreamCreate(0))

    def get_engine_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns TensorRT engine attributes.'
        return (self.engine, self.context, self.context_lock, self.inputs, self.outputs, self.gpu_allocations, self.cpu_allocations, self.stream)
TensorRTInferenceFn = Callable[[Sequence[np.ndarray], TensorRTEngine, Optional[Dict[str, Any]]], Iterable[PredictionResult]]

def _default_tensorRT_inference_fn(batch: Sequence[np.ndarray], engine: TensorRTEngine, inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
    if False:
        while True:
            i = 10
    from cuda import cuda
    (engine, context, context_lock, inputs, outputs, gpu_allocations, cpu_allocations, stream) = engine.get_engine_attrs()
    with context_lock:
        _assign_or_fail(cuda.cuMemcpyHtoDAsync(inputs[0]['allocation'], np.ascontiguousarray(batch), inputs[0]['size'], stream))
        context.execute_async_v2(gpu_allocations, stream)
        for output in range(len(cpu_allocations)):
            _assign_or_fail(cuda.cuMemcpyDtoHAsync(cpu_allocations[output], outputs[output]['allocation'], outputs[output]['size'], stream))
        _assign_or_fail(cuda.cuStreamSynchronize(stream))
        predictions = []
        for idx in range(len(batch)):
            predictions.append([prediction[idx] for prediction in cpu_allocations])
        return utils._convert_to_result(batch, predictions)

class TensorRTEngineHandlerNumPy(ModelHandler[np.ndarray, PredictionResult, TensorRTEngine]):

    def __init__(self, min_batch_size: int, max_batch_size: int, *, inference_fn: TensorRTInferenceFn=_default_tensorRT_inference_fn, large_model: bool=False, **kwargs):
        if False:
            print('Hello World!')
        'Implementation of the ModelHandler interface for TensorRT.\n\n    Example Usage::\n\n      pcoll | RunInference(\n          TensorRTEngineHandlerNumPy(\n            min_batch_size=1,\n            max_batch_size=1,\n            engine_path="my_uri"))\n\n    **NOTE:** This API and its implementation are under development and\n    do not provide backward compatibility guarantees.\n\n    Args:\n      min_batch_size: minimum accepted batch size.\n      max_batch_size: maximum accepted batch size.\n      inference_fn: the inference function to use on RunInference calls.\n        default: _default_tensorRT_inference_fn\n      large_model: set to true if your model is large enough to run into\n        memory pressure if you load multiple copies. Given a model that\n        consumes N memory and a machine with W cores and M memory, you should\n        set this to True if N*W > M.\n      kwargs: Additional arguments like \'engine_path\' and \'onnx_path\' are\n        currently supported. \'env_vars\' can be used to set environment variables\n        before loading the model.\n\n    See https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/\n    for details\n    '
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.inference_fn = inference_fn
        if 'engine_path' in kwargs:
            self.engine_path = kwargs.get('engine_path')
        elif 'onnx_path' in kwargs:
            self.onnx_path = kwargs.get('onnx_path')
        self._env_vars = kwargs.get('env_vars', {})
        self._large_model = large_model

    def batch_elements_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Sets min_batch_size and max_batch_size of a TensorRT engine.'
        return {'min_batch_size': self.min_batch_size, 'max_batch_size': self.max_batch_size}

    def load_model(self) -> TensorRTEngine:
        if False:
            i = 10
            return i + 15
        'Loads and initializes a TensorRT engine for processing.'
        engine = _load_engine(self.engine_path)
        return TensorRTEngine(engine)

    def load_onnx(self) -> Tuple[trt.INetworkDefinition, trt.Builder]:
        if False:
            return 10
        'Loads and parses an onnx model for processing.'
        return _load_onnx(self.onnx_path)

    def build_engine(self, network: trt.INetworkDefinition, builder: trt.Builder) -> TensorRTEngine:
        if False:
            print('Hello World!')
        'Build an engine according to parsed/created network.'
        engine = _build_engine(network, builder)
        return TensorRTEngine(engine)

    def run_inference(self, batch: Sequence[np.ndarray], engine: TensorRTEngine, inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            return 10
        '\n    Runs inferences on a batch of Tensors and returns an Iterable of\n    TensorRT Predictions.\n\n    Args:\n      batch: A np.ndarray or a np.ndarray that represents a concatenation\n        of multiple arrays as a batch.\n      engine: A TensorRT engine.\n      inference_args: Any additional arguments for an inference\n        that are not applicable to TensorRT.\n\n    Returns:\n      An Iterable of type PredictionResult.\n    '
        return self.inference_fn(batch, engine, inference_args)

    def get_num_bytes(self, batch: Sequence[np.ndarray]) -> int:
        if False:
            i = 10
            return i + 15
        '\n    Returns:\n      The number of bytes of data for a batch of Tensors.\n    '
        return sum((np_array.itemsize for np_array in batch))

    def get_metrics_namespace(self) -> str:
        if False:
            while True:
                i = 10
        '\n    Returns a namespace for metrics collected by the RunInference transform.\n    '
        return 'BeamML_TensorRT'

    def share_model_across_processes(self) -> bool:
        if False:
            print('Hello World!')
        return self._large_model
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
import numpy
import onnx
import onnxruntime as ort
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference import utils
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult
__all__ = ['OnnxModelHandlerNumpy']
NumpyInferenceFn = Callable[[Sequence[numpy.ndarray], ort.InferenceSession, Optional[Dict[str, Any]]], Iterable[PredictionResult]]

def default_numpy_inference_fn(inference_session: ort.InferenceSession, batch: Sequence[numpy.ndarray], inference_args: Optional[Dict[str, Any]]=None) -> Any:
    if False:
        return 10
    ort_inputs = {inference_session.get_inputs()[0].name: numpy.stack(batch, axis=0)}
    if inference_args:
        ort_inputs = {**ort_inputs, **inference_args}
    ort_outs = inference_session.run(None, ort_inputs)[0]
    return ort_outs

class OnnxModelHandlerNumpy(ModelHandler[numpy.ndarray, PredictionResult, ort.InferenceSession]):

    def __init__(self, model_uri: str, session_options=None, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], provider_options=None, *, inference_fn: NumpyInferenceFn=default_numpy_inference_fn, large_model: bool=False, **kwargs):
        if False:
            return 10
        ' Implementation of the ModelHandler interface for onnx\n    using numpy arrays as input.\n    Note that inputs to ONNXModelHandler should be of the same sizes\n\n    Example Usage::\n\n      pcoll | RunInference(OnnxModelHandler(model_uri="my_uri"))\n\n    Args:\n      model_uri: The URI to where the model is saved.\n      inference_fn: The inference function to use on RunInference calls.\n        default=default_numpy_inference_fn\n      large_model: set to true if your model is large enough to run into\n        memory pressure if you load multiple copies. Given a model that\n        consumes N memory and a machine with W cores and M memory, you should\n        set this to True if N*W > M.\n      kwargs: \'env_vars\' can be used to set environment variables\n        before loading the model.\n    '
        self._model_uri = model_uri
        self._session_options = session_options
        self._providers = providers
        self._provider_options = provider_options
        self._model_inference_fn = inference_fn
        self._env_vars = kwargs.get('env_vars', {})
        self._large_model = large_model

    def load_model(self) -> ort.InferenceSession:
        if False:
            for i in range(10):
                print('nop')
        'Loads and initializes an onnx inference session for processing.'
        f = FileSystems.open(self._model_uri, 'rb')
        model_proto = onnx.load(f)
        model_proto_bytes = onnx._serialize(model_proto)
        ort_session = ort.InferenceSession(model_proto_bytes, sess_options=self._session_options, providers=self._providers, provider_options=self._provider_options)
        return ort_session

    def run_inference(self, batch: Sequence[numpy.ndarray], inference_session: ort.InferenceSession, inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            return 10
        'Runs inferences on a batch of numpy arrays.\n\n    Args:\n      batch: A sequence of examples as numpy arrays. They should\n        be single examples.\n      inference_session: An onnx inference session.\n        Must be runnable with input x where x is sequence of numpy array\n      inference_args: Any additional arguments for an inference.\n\n    Returns:\n      An Iterable of type PredictionResult.\n    '
        predictions = self._model_inference_fn(inference_session, batch, inference_args)
        return utils._convert_to_result(batch, predictions)

    def get_num_bytes(self, batch: Sequence[numpy.ndarray]) -> int:
        if False:
            while True:
                i = 10
        '\n    Returns:\n      The number of bytes of data for a batch.\n    '
        return sum((np_array.itemsize for np_array in batch))

    def get_metrics_namespace(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n    Returns:\n       A namespace for metrics collected by the RunInference transform.\n    '
        return 'BeamML_Onnx'

    def share_model_across_processes(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._large_model
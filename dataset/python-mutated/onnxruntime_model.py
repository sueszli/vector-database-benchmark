from pathlib import Path
import onnxruntime as ort
import onnx
from bigdl.nano.utils.common import invalidInputError, _flatten
import numpy as np

class ONNXRuntimeModel:

    def __init__(self, onnx_filepath, session_options=None):
        if False:
            return 10
        self.onnx_filepath = onnx_filepath
        self.onnx_model = None
        self.ortsess = None
        self.session_options = session_options
        self._build_ortsess(session_options)

    def forward_step(self, *inputs, **kwargs):
        if False:
            print('Hello World!')
        '\n        This function run through the onnxruntime forwarding step\n        '
        flattened_inputs = []
        _flatten(inputs, flattened_inputs)
        zipped_inputs = dict(zip(self.forward_args, flattened_inputs))
        if kwargs is not None and len(kwargs) > 0:
            zipped_inputs.update(kwargs)
        if len(self._forward_args) != len(zipped_inputs):
            invalidInputError(False, f"The length of inputs is inconsistent with the length of ONNX Runtime session's inputs, got model_forward_args: {self._forward_args}, and flattened inputs: {flattened_inputs}")
        ort_outs = self.ortsess.run(None, zipped_inputs)
        return ort_outs

    @property
    def forward_args(self):
        if False:
            while True:
                i = 10
        return self._forward_args

    def _build_ortsess(self, sess_options=None):
        if False:
            return 10
        '\n        Internal function to build a ortsess.\n\n        :param sess_options: ortsess options in ort.SessionOptions type\n        '
        onnx_path_or_bytes = self.onnx_filepath
        if isinstance(self.onnx_filepath, str):
            self.onnx_model = onnx.load(self.onnx_filepath)
        elif isinstance(self.onnx_filepath, bytes):
            self.onnx_model = onnx.load_model_from_string(self.onnx_filepath)
        else:
            invalidInputError(isinstance(self.onnx_filepath, onnx.ModelProto), errMsg='Model type {} is not a legal ONNX model.'.format(type(self.onnx_filepath)))
            self.onnx_model = self.onnx_filepath
            onnx_path_or_bytes = self.onnx_filepath.SerializeToString()
        self.ortsess = ort.InferenceSession(onnx_path_or_bytes, sess_options=sess_options)
        self._forward_args = list(map(lambda x: x.name, self.ortsess.get_inputs()))

    def _save_model(self, path, compression='fp32'):
        if False:
            while True:
                i = 10
        '\n        Save ONNXRuntimeModel to local as an onnx file\n\n        :param path: Path to save the model.\n        '
        path = Path(path)
        invalidInputError(self.onnx_model, "self.ie_network shouldn't be None.")
        invalidInputError(path.suffix == '.onnx', "Path of onnx model must be with '.onnx' suffix.")
        onnx.save(self.onnx_model, str(path))
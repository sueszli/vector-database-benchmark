from __future__ import absolute_import as _
from __future__ import division as _
from __future__ import print_function as _
import numpy as np
from typing import Any, Sequence, List
from onnx.backend.base import BackendRep, namedtupledict
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from coremltools.proto import FeatureTypes_pb2 as ft
from coremltools.models import MLModel
from typing import Dict, Any, Text, Tuple
from onnx import TensorProto
from ._graph import EdgeInfo
from ._converter import SupportedVersion

def _set_dtypes(input_dict, model):
    if False:
        return 10
    spec = model.get_spec()
    for input_ in spec.description.input:
        if input_.type.HasField('multiArrayType') and input_.name in input_dict:
            if input_.type.multiArrayType.dataType == ft.ArrayFeatureType.INT32:
                input_dict[input_.name] = input_dict[input_.name].astype(np.int32)
            if input_.type.multiArrayType.dataType == ft.ArrayFeatureType.FLOAT32:
                input_dict[input_.name] = input_dict[input_.name].astype(np.float32)
            if input_.type.multiArrayType.dataType == ft.ArrayFeatureType.DOUBLE:
                input_dict[input_.name] = input_dict[input_.name].astype(np.float64)

class CoreMLRep(BackendRep):

    def __init__(self, coreml_model, onnx_outputs_info, useCPUOnly=False, minimum_ios_deployment_target='12'):
        if False:
            while True:
                i = 10
        super(CoreMLRep, self).__init__()
        self.model = coreml_model
        self.useCPUOnly = useCPUOnly
        self.minimum_ios_deployment_target = minimum_ios_deployment_target
        spec = coreml_model.get_spec()
        self.input_names = [str(i.name) for i in spec.description.input]
        self.output_names = [str(o.name) for o in spec.description.output]
        self.onnx_outputs_info = onnx_outputs_info

    def run(self, inputs, **kwargs):
        if False:
            while True:
                i = 10
        super(CoreMLRep, self).run(inputs, **kwargs)
        inputs_ = inputs
        _reshaped = False
        if not SupportedVersion.is_nd_array_supported(self.minimum_ios_deployment_target):
            for (i, input_) in enumerate(inputs_):
                shape = input_.shape
                if len(shape) == 4 or len(shape) == 2:
                    inputs_[i] = input_[np.newaxis, :]
                    _reshaped = True
                elif len(shape) == 3:
                    spec = self.model.get_spec()
                    spec_shape = [int(k) for k in spec.description.input[i].type.multiArrayType.shape]
                    prod = spec_shape[0] * spec_shape[1] * spec_shape[2]
                    onnx_shape = list(shape)
                    if onnx_shape != spec_shape:
                        if onnx_shape[2] == prod:
                            inputs_[i] = np.reshape(inputs_[i], [onnx_shape[0], onnx_shape[1]] + spec_shape)
                        elif onnx_shape[1] * onnx_shape[2] == prod:
                            inputs_[i] = np.reshape(inputs_[i], [1, onnx_shape[0]] + spec_shape)
        input_dict = dict(zip(self.input_names, map(np.array, inputs_)))
        _set_dtypes(input_dict, self.model)
        prediction = self.model.predict(input_dict, self.useCPUOnly)
        output_values = [prediction[name] for name in self.output_names]
        if not SupportedVersion.is_nd_array_supported(self.minimum_ios_deployment_target):
            for (i, output_) in enumerate(output_values):
                shape = output_.shape
                try:
                    output_values[i] = np.reshape(output_, self.onnx_outputs_info[self.output_names[i]][2])
                except RuntimeError:
                    print("Output '%s' shape incompatible between CoreML (%s) and onnx (%s)" % (self.output_names[i], output_.shape, self.onnx_outputs_info[self.output_names[i]]))
        for (i, output_) in enumerate(output_values):
            output_type = self.onnx_outputs_info[self.output_names[i]][1]
            if TENSOR_TYPE_TO_NP_TYPE[output_type] != output_values[i].dtype:
                output_values[i] = output_values[i].astype(TENSOR_TYPE_TO_NP_TYPE[output_type])
        result = namedtupledict('Outputs', self.output_names)(*output_values)
        return result
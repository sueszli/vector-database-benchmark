"""
Machine learning model to ONNX export module
"""
from ..base import Pipeline
try:
    from onnxmltools import convert_sklearn
    from onnxmltools.convert.common.data_types import StringTensorType
    from skl2onnx.helpers.onnx_helper import save_onnx_model, select_model_inputs_outputs
    ONNX_MLTOOLS = True
except ImportError:
    ONNX_MLTOOLS = False

class MLOnnx(Pipeline):
    """
    Exports a machine learning model to ONNX using ONNXMLTools.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Creates a new MLOnnx pipeline.\n        '
        if not ONNX_MLTOOLS:
            raise ImportError('MLOnnx pipeline is not available - install "pipeline" extra to enable')

    def __call__(self, model, task='default', output=None, opset=12):
        if False:
            i = 10
            return i + 15
        '\n        Exports a machine learning model to ONNX using ONNXMLTools.\n\n        Args:\n            model: model to export\n            task: optional model task or category\n            output: optional output model path, defaults to return byte array if None\n            opset: onnx opset, defaults to 12\n\n        Returns:\n            path to model output or model as bytes depending on output parameter\n        '
        model = convert_sklearn(model, task, initial_types=[('input_ids', StringTensorType([None, None]))], target_opset=opset)
        model = select_model_inputs_outputs(model, outputs='probabilities')
        model.graph.output[0].name = 'logits'
        for node in model.graph.node:
            if node.output[0] == 'probabilities':
                node.output[0] = 'logits'
        model = save_onnx_model(model, output)
        return output if output else model
"""
Hugging Face Transformers ONNX export module
"""
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from tempfile import NamedTemporaryFile
try:
    from onnxruntime.quantization import quantize_dynamic
    ONNX_RUNTIME = True
except ImportError:
    ONNX_RUNTIME = False
from torch import nn
from torch.onnx import export
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
from ...models import PoolingFactory
from ..tensors import Tensors

class HFOnnx(Tensors):
    """
    Exports a Hugging Face Transformer model to ONNX.
    """

    def __call__(self, path, task='default', output=None, quantize=False, opset=12):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exports a Hugging Face Transformer model to ONNX.\n\n        Args:\n            path: path to model, accepts Hugging Face model hub id, local path or (model, tokenizer) tuple\n            task: optional model task or category, determines the model type and outputs, defaults to export hidden state\n            output: optional output model path, defaults to return byte array if None\n            quantize: if model should be quantized (requires onnx to be installed), defaults to False\n            opset: onnx opset, defaults to 12\n\n        Returns:\n            path to model output or model as bytes depending on output parameter\n        '
        (inputs, outputs, model) = self.parameters(task)
        if isinstance(path, (list, tuple)):
            (model, tokenizer) = path
            model = model.cpu()
        else:
            model = model(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
        dummy = dict(tokenizer(['test inputs'], return_tensors='pt'))
        output = output if output else BytesIO()
        export(model, (dummy,), output, opset_version=opset, do_constant_folding=True, input_names=list(inputs.keys()), output_names=list(outputs.keys()), dynamic_axes=dict(chain(inputs.items(), outputs.items())))
        if quantize:
            if not ONNX_RUNTIME:
                raise ImportError('onnxruntime is not available - install "pipeline" extra to enable')
            output = self.quantization(output)
        if isinstance(output, BytesIO):
            output.seek(0)
            output = output.read()
        return output

    def quantization(self, output):
        if False:
            print('Hello World!')
        '\n        Quantizes an ONNX model.\n\n        Args:\n            output: path to ONNX model or BytesIO with model data\n\n        Returns:\n            quantized model as file path or bytes\n        '
        temp = None
        if isinstance(output, BytesIO):
            with NamedTemporaryFile(suffix='.quant', delete=False) as tmpfile:
                temp = tmpfile.name
            with open(temp, 'wb') as f:
                f.write(output.getbuffer())
            output = temp
        quantize_dynamic(output, output, extra_options={'MatMulConstBOnly': False})
        if temp:
            with open(temp, 'rb') as f:
                output = f.read()
        return output

    def parameters(self, task):
        if False:
            i = 10
            return i + 15
        '\n        Defines inputs and outputs for an ONNX model.\n\n        Args:\n            task: task name used to lookup model configuration\n\n        Returns:\n            (inputs, outputs, model function)\n        '
        inputs = OrderedDict([('input_ids', {0: 'batch', 1: 'sequence'}), ('attention_mask', {0: 'batch', 1: 'sequence'}), ('token_type_ids', {0: 'batch', 1: 'sequence'})])
        config = {'default': (OrderedDict({'last_hidden_state': {0: 'batch', 1: 'sequence'}}), AutoModel.from_pretrained), 'pooling': (OrderedDict({'embeddings': {0: 'batch', 1: 'sequence'}}), lambda x: PoolingOnnx(x, -1)), 'question-answering': (OrderedDict({'start_logits': {0: 'batch', 1: 'sequence'}, 'end_logits': {0: 'batch', 1: 'sequence'}}), AutoModelForQuestionAnswering.from_pretrained), 'text-classification': (OrderedDict({'logits': {0: 'batch'}}), AutoModelForSequenceClassification.from_pretrained)}
        config['zero-shot-classification'] = config['text-classification']
        return (inputs,) + config[task]

class PoolingOnnx(nn.Module):
    """
    Extends Pooling methods to name inputs to model, which is required to export to ONNX.
    """

    def __init__(self, path, device):
        if False:
            return 10
        '\n        Creates a new PoolingOnnx instance.\n\n        Args:\n            path: path to model, accepts Hugging Face model hub id or local path\n            device: tensor device id\n        '
        super().__init__()
        self.model = PoolingFactory.create({'path': path, 'device': device})

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        if False:
            i = 10
            return i + 15
        '\n        Runs inputs through pooling model and returns outputs.\n\n        Args:\n            inputs: model inputs\n\n        Returns:\n            model outputs\n        '
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            inputs['token_type_ids'] = token_type_ids
        return self.model.forward(**inputs)
"""
Hugging Face Transformers pipeline wrapper module
"""
import inspect
from transformers import pipeline
from ..models import Models
from ..util import Resolver
from .tensors import Tensors

class HFPipeline(Tensors):
    """
    Light wrapper around Hugging Face Transformers pipeline component for selected tasks. Adds support for model
    quantization and minor interface changes.
    """

    def __init__(self, task, path=None, quantize=False, gpu=False, model=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Loads a new pipeline model.\n\n        Args:\n            task: pipeline task or category\n            path: optional path to model, accepts Hugging Face model hub id, local path or (model, tokenizer) tuple.\n                  uses default model for task if not provided\n            quantize: if model should be quantized, defaults to False\n            gpu: True/False if GPU should be enabled, also supports a GPU device id\n            model: optional existing pipeline model to wrap\n            kwargs: additional keyword arguments to pass to pipeline model\n        '
        if model:
            self.pipeline = model.pipeline if isinstance(model, HFPipeline) else model
        else:
            deviceid = Models.deviceid(gpu) if 'device_map' not in kwargs else None
            device = Models.device(deviceid) if deviceid is not None else None
            (modelargs, kwargs) = self.parseargs(**kwargs)
            if isinstance(path, (list, tuple)):
                config = path[1] if path[1] and isinstance(path[1], str) else None
                model = Models.load(path[0], config, task)
                self.pipeline = pipeline(task, model=model, tokenizer=path[1], device=device, model_kwargs=modelargs, **kwargs)
            else:
                self.pipeline = pipeline(task, model=path, device=device, model_kwargs=modelargs, **kwargs)
            if deviceid == -1 and quantize:
                self.pipeline.model = self.quantize(self.pipeline.model)
        Models.checklength(self.pipeline.model, self.pipeline.tokenizer)

    def parseargs(self, **kwargs):
        if False:
            return 10
        '\n        Inspects the pipeline method and splits kwargs into model args and pipeline args.\n\n        Args:\n            kwargs: all keyword arguments\n\n        Returns:\n            (model args, pipeline args)\n        '
        args = inspect.getfullargspec(pipeline).args
        dtype = kwargs.get('torch_dtype')
        if dtype and isinstance(dtype, str) and (dtype != 'auto'):
            kwargs['torch_dtype'] = Resolver()(dtype)
        return ({arg: value for (arg, value) in kwargs.items() if arg not in args}, {arg: value for (arg, value) in kwargs.items() if arg in args})

    def maxlength(self):
        if False:
            print('Hello World!')
        '\n        Gets the max length to use for generate calls.\n\n        Returns:\n            max length\n        '
        return Models.maxlength(self.pipeline.model, self.pipeline.tokenizer)
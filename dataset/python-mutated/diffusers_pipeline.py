import os
from typing import Any, Dict, Generator, List, Union
from modelscope.pipelines.base import Input, Pipeline
from modelscope.utils.constant import Hubs
from modelscope.utils.device import create_device
from modelscope.utils.hub import snapshot_download

class DiffusersPipeline(Pipeline):

    def __init__(self, model: str, device: str='gpu', **kwargs):
        if False:
            print('Hello World!')
        "\n        use `model` to create a diffusers pipeline\n        Args:\n            model: model id on modelscope hub or local dir.\n            device: str = 'gpu'\n        "
        self.device_name = device
        self.cfg = None
        self.preprocessor = None
        self.framework = None
        self.device = create_device(self.device_name)
        self.hubs = kwargs.get('hubs', Hubs.modelscope)
        model_folder = model
        if not os.path.isdir(model_folder):
            if self.hubs != Hubs.modelscope:
                raise NotImplementedError('Only support model retrieval from ModelScope hub for now.')
            model_folder = snapshot_download(model)
        self.model = model_folder
        self.models = [self.model]
        self.has_multiple_models = len(self.models) > 1

    def preprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return inputs

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return inputs

    def __call__(self, input: Union[Input, List[Input]], *args, **kwargs) -> Union[Dict[str, Any], Generator]:
        if False:
            while True:
                i = 10
        (preprocess_params, forward_params, postprocess_params) = self._sanitize_parameters(**kwargs)
        self._check_input(input)
        out = self.preprocess(input, **preprocess_params)
        out = self.forward(out, **forward_params)
        out = self.postprocess(out, **postprocess_params)
        self._check_output(out)
        return out
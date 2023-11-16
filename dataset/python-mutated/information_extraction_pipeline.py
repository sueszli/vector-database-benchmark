from typing import Any, Dict, Optional, Union
import torch
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModelFile, Tasks
__all__ = ['InformationExtractionPipeline']

@PIPELINES.register_module(Tasks.information_extraction, module_name=Pipelines.relation_extraction)
@PIPELINES.register_module(Tasks.relation_extraction, module_name=Pipelines.relation_extraction)
class InformationExtractionPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], preprocessor: Optional[Preprocessor]=None, config_file: str=None, device: str='gpu', auto_collate=True, sequence_length=512, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n\n        Args:\n            model (str or Model): Supply either a local model dir which supported information extraction task, or a\n            model id from the model hub, or a torch model instance.\n            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for\n            the model if supplied.\n            kwargs (dict, `optional`):\n                Extra kwargs passed into the preprocessor's constructor.\n        "
        super().__init__(model=model, preprocessor=preprocessor, config_file=config_file, device=device, auto_collate=auto_collate)
        assert isinstance(self.model, Model), f'please check whether model config exists in {ModelFile.CONFIGURATION}'
        if self.preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(self.model.model_dir, sequence_length=sequence_length, **kwargs)
        self.model.eval()

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        with torch.no_grad():
            return self.model(**inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any], **postprocess_params) -> Dict[str, str]:
        if False:
            return 10
        return inputs
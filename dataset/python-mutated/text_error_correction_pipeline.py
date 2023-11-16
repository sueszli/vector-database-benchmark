from typing import Any, Dict, Optional, Union
import torch
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import BartForTextErrorCorrection
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModelFile, Tasks
__all__ = ['TextErrorCorrectionPipeline']

@PIPELINES.register_module(Tasks.text_error_correction, module_name=Pipelines.text_error_correction)
class TextErrorCorrectionPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], preprocessor: Optional[Preprocessor]=None, config_file: str=None, device: str='gpu', auto_collate=True, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Use `model` and `preprocessor` to create a nlp text correction pipeline.\n\n        Args:\n            model (BartForTextErrorCorrection): A model instance, or a model local dir, or a model id in the model hub.\n            preprocessor (TextErrorCorrectionPreprocessor): An optional preprocessor instance.\n            kwargs (dict, `optional`):\n                Extra kwargs passed into the preprocessor's constructor.\n\n        Examples:\n            >>> from modelscope.pipelines import pipeline\n            >>> pipeline_ins = pipeline(\n            >>>    task='text-error-correction', model='damo/nlp_bart_text-error-correction_chinese')\n            >>> sentence1 = '随着中国经济突飞猛近，建造工业与日俱增'\n            >>> print(pipeline_ins(sentence1))\n\n        To view other examples plese check tests/pipelines/test_text_error_correction.py.\n        "
        super().__init__(model=model, preprocessor=preprocessor, config_file=config_file, device=device, auto_collate=auto_collate)
        assert isinstance(self.model, Model), f'please check whether model config exists in {ModelFile.CONFIGURATION}'
        if preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(self.model.model_dir, **kwargs)
        self.vocab = self.preprocessor.vocab

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Tensor], **postprocess_params) -> Dict[str, str]:
        if False:
            return 10
        "\n        Args:\n            inputs (Dict[str, Tensor])\n            Examples:\n                {\n                    'predictions': Tensor([1377, 4959, 2785, 6392...]), # tokens need to be decode by tokenizer\n                }\n        Returns:\n            Dict[str, str]: which contains following:\n                - 'output': output str, for example '随着中国经济突飞猛进，建造工业与日俱增'\n\n        "
        sc_tensor = inputs['predictions']
        if isinstance(sc_tensor, list):
            sc_tensor = sc_tensor[0]
        sc_sent = self.vocab.string(sc_tensor, extra_symbols_to_ignore={self.vocab.pad()})
        sc_sent = (sc_sent + ' ').replace('##', '').rstrip()
        sc_sent = ''.join(sc_sent.split())
        return {OutputKeys.OUTPUT: sc_sent}
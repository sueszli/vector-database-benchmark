import os
from typing import Any, Dict
import json
from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.preprocessors.nlp import IntentBPETextField
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModelFile
from modelscope.utils.type_assert import type_assert
__all__ = ['DialogIntentPredictionPreprocessor']

@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.dialog_intent_preprocessor)
class DialogIntentPredictionPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'preprocess the data\n\n        Args:\n            model_dir (str): model path\n        '
        super().__init__(*args, **kwargs)
        self.model_dir: str = model_dir
        self.config = Config.from_file(os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        self.text_field = IntentBPETextField(self.model_dir, config=self.config)
        self.categories = None
        with open(os.path.join(self.model_dir, 'categories.json'), 'r', encoding='utf-8') as f:
            self.categories = json.load(f)
        assert len(self.categories) == 77

    @type_assert(object, str)
    def __call__(self, data: str) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        "process the raw input data\n\n        Args:\n            data (str): a sentence\n                Example:\n                    'What do I need to do for the card activation?'\n\n        Returns:\n            Dict[str, Any]: the preprocessed data\n                Example:\n                    {\n                        'src_token': array([[13,  2054,  2079,  1045...]]),\n                        'src_pos': array([[ 0,  1,  2,  3...]]),\n                        'src_type': array([[1, 1, 1, 1...]]),\n                        'src_turn': array([[1, 1, 1, 1...]]),\n                        'src_mask': array([[1, 1, 1, 1...]]),\n                        'mlm_token': array([[13,  2054,  2079,  1045...]]),\n                        'mlm_label': array([[0, 0, 0, 0...]]),\n                        'mlm_mask': array([[0, 0, 0, 0...]]),\n                        'tgt_token': array([[29, 30, 31, 32...]]),\n                        'tgt_mask': array([[1, 1, 1, 1...]]),\n                        'ids': array([0]),\n                        'intent_label': array([-1])\n                    }\n        "
        samples = self.text_field.preprocessor([data])
        (samples, _) = self.text_field.collate_fn_multi_turn(samples)
        return samples
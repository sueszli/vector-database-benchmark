import os
from typing import Any, Dict
from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.preprocessors.nlp import MultiWOZBPETextField
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModelFile
from modelscope.utils.type_assert import type_assert
__all__ = ['DialogModelingPreprocessor']

@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.dialog_modeling_preprocessor)
class DialogModelingPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'preprocess the data\n\n        Args:\n            model_dir (str): model path\n        '
        super().__init__(*args, **kwargs)
        self.model_dir: str = model_dir
        self.config = Config.from_file(os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        import torch
        self.config.use_gpu = self.config.use_gpu and torch.cuda.is_available()
        self.text_field = MultiWOZBPETextField(config=self.config, model_dir=self.model_dir)

    @type_assert(object, Dict)
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        "process the raw input data\n\n        Args:\n            data (Dict[str, Any]): A sentence and dialogue history info.\n                Example:\n                    {\n                        'user_input': 'i want to leave after 17:15 .',\n                        'history': {\n                            'labels': [[13, 1045, 2052, 2066...]],\n                            'resp': [14, 1045, 2064, 2393...],\n                            'bspn': [15, 43, 7688, 10733...],\n                            'db': [19, 24, 20],\n                            'aspn': [16, 43, 48, 2681, 7180, 10],\n                            'output': ['i', 'can', 'help', 'with'...]\n                        }\n                    }\n\n        Returns:\n            Dict[str, Any]: the preprocessed data\n        "
        import torch
        first_turn = True if len(data['history']) == 0 else False
        user_ids = self.text_field.get_ids(data['user_input'])
        (inputs, prompt_id) = self.text_field.convert_turn_eval(turn={'user': user_ids}, pv_turn=data['history'], first_turn=first_turn)
        (batch, batch_size) = self.text_field.collate_fn_multi_turn(samples=[inputs])
        data['first_turn'] = first_turn
        data['batch'] = batch
        data['batch_size'] = batch_size
        data['prompt_id'] = prompt_id
        data['labels'] = [torch.Tensor(item).int() for item in inputs['labels']]
        return data
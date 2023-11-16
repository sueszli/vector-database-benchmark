import os
from typing import Dict
from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.models.nlp.space import SpaceGenerator, SpaceModelBase
from modelscope.preprocessors.nlp import MultiWOZBPETextField
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
__all__ = ['SpaceForDialogModeling']

@MODELS.register_module(Tasks.task_oriented_conversation, module_name=Models.space_modeling)
class SpaceForDialogModeling(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'initialize the test generation model from the `model_dir` path.\n\n        Args:\n            model_dir (`str`):\n                The model path.\n            text_field (`BPETextField`, *optional*, defaults to `MultiWOZBPETextField`):\n                The text field.\n            config (`Config`, *optional*, defaults to config in model hub):\n                The config.\n        '
        super().__init__(model_dir, *args, **kwargs)
        from modelscope.trainers.nlp.space.trainer.gen_trainer import MultiWOZTrainer
        self.model_dir = model_dir
        self.config = kwargs.pop('config', Config.from_file(os.path.join(self.model_dir, ModelFile.CONFIGURATION)))
        import torch
        self.config.use_gpu = True if ('device' not in kwargs or kwargs['device'] == 'gpu') and torch.cuda.is_available() else False
        self.text_field = kwargs.pop('text_field', MultiWOZBPETextField(config=self.config, model_dir=self.model_dir))
        self.generator = SpaceGenerator.create(self.config, reader=self.text_field)
        self.model = SpaceModelBase.create(model_dir=model_dir, config=self.config, reader=self.text_field, generator=self.generator)

        def to_tensor(array):
            if False:
                print('Hello World!')
            '\n            numpy array -> tensor\n            '
            import torch
            array = torch.tensor(array)
            return array.cuda() if self.config.use_gpu else array
        self.trainer = MultiWOZTrainer(model=self.model, to_tensor=to_tensor, config=self.config, reader=self.text_field, evaluator=None)
        self.trainer.load()

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if False:
            return 10
        "return the result by the model\n\n        Args:\n            input (Dict[str, Tensor]): the preprocessed data\n\n        Returns:\n            Dict[str, Tensor]: results\n                Example:\n                    {\n                        'labels': array([1,192,321,12]), # lable\n                        'resp': array([293,1023,123,1123]), #vocab label for response\n                        'bspn': array([123,321,2,24,1 ]),\n                        'aspn': array([47,8345,32,29,1983]),\n                        'db': array([19, 24, 20]),\n                    }\n\n        Examples:\n            >>> from modelscope.hub.snapshot_download import snapshot_download\n            >>> from modelscope.models.nlp import SpaceForDialogModeling\n            >>> from modelscope.preprocessors import DialogModelingPreprocessor\n            >>> cache_path = snapshot_download('damo/nlp_space_dialog-modeling')\n            >>> preprocessor = DialogModelingPreprocessor(model_dir=cache_path)\n            >>> model = SpaceForDialogModeling(model_dir=cache_path,\n                    text_field=preprocessor.text_field,\n                    config=preprocessor.config)\n            >>> print(model(preprocessor({\n                    'user_input': 'i would like a taxi from saint john 's college to pizza hut fen ditton .',\n                    'history': {}\n                })))\n        "
        first_turn = input['first_turn']
        batch = input['batch']
        prompt_id = input['prompt_id']
        labels = input['labels']
        old_pv_turn = input['history']
        pv_turn = self.trainer.forward(first_turn=first_turn, batch=batch, prompt_id=prompt_id, labels=labels, old_pv_turn=old_pv_turn)
        return pv_turn
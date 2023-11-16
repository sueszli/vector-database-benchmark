import os
from typing import Dict
from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.models.nlp.space import SpaceGenerator, SpaceModelBase
from modelscope.preprocessors.nlp import IntentBPETextField
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
__all__ = ['SpaceForDialogIntent']

@MODELS.register_module(Tasks.task_oriented_conversation, module_name=Models.space_intent)
class SpaceForDialogIntent(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'initialize the test generation model from the `model_dir` path.\n\n        Args:\n            model_dir (str): the model path.\n            text_field (`BPETextField`, *optional*, defaults to `IntentBPETextField`):\n                The text field.\n            config (`Config`, *optional*, defaults to config in model hub):\n                The config.\n        '
        super().__init__(model_dir, *args, **kwargs)
        from modelscope.trainers.nlp.space.trainer.intent_trainer import IntentTrainer
        self.model_dir = model_dir
        self.config = kwargs.pop('config', Config.from_file(os.path.join(self.model_dir, ModelFile.CONFIGURATION)))
        self.text_field = kwargs.pop('text_field', IntentBPETextField(self.model_dir, config=self.config))
        self.generator = SpaceGenerator.create(self.config, reader=self.text_field)
        self.model = SpaceModelBase.create(model_dir=model_dir, config=self.config, reader=self.text_field, generator=self.generator)

        def to_tensor(array):
            if False:
                return 10
            '\n            numpy array -> tensor\n            '
            import torch
            array = torch.tensor(array)
            return array.cuda() if self.config.use_gpu else array
        self.trainer = IntentTrainer(model=self.model, to_tensor=to_tensor, config=self.config, reader=self.text_field)
        self.trainer.load()

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if False:
            while True:
                i = 10
        'return the result by the model\n\n        Args:\n            input (Dict[str, Tensor]): the preprocessed data\n\n        Returns:\n            Dict[str, Tensor]: results\n                Example:\n                    {\n                        \'pred\': array([2.62349960e-03 4.12110658e-03 4.12748595e-05 3.77560973e-05\n                                1.08599677e-04 1.72710388e-05 2.95618793e-05 1.93638436e-04\n                                6.45841064e-05 1.15997791e-04 5.11605394e-05 9.87020373e-01\n                                2.66957268e-05 4.72324500e-05 9.74208378e-05], dtype=float32),\n                    }\n        Example:\n            >>> from modelscope.hub.snapshot_download import snapshot_download\n            >>> from modelscope.models.nlp import SpaceForDialogIntent\n            >>> from modelscope.preprocessors import DialogIntentPredictionPreprocessor\n            >>> cache_path = snapshot_download(\'damo/nlp_space_dialog-intent-prediction\')\n            >>> preprocessor = DialogIntentPredictionPreprocessor(model_dir=cache_path)\n            >>> model = SpaceForDialogIntent(\n                    model_dir=cache_path,\n                    text_field=preprocessor.text_field,\n                    config=preprocessor.config)\n            >>> print(model(preprocessor("What do I need to do for the card activation?")))\n        '
        import numpy as np
        pred = self.trainer.forward(input)
        pred = np.squeeze(pred[0], 0)
        return {'pred': pred}
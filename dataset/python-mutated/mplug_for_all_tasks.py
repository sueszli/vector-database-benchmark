import os.path as osp
from typing import Dict, List
from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
__all__ = ['MPlugForAllTasks', 'HiTeAForAllTasks']

@MODELS.register_module(Tasks.visual_question_answering, module_name=Models.mplug)
@MODELS.register_module(Tasks.image_captioning, module_name=Models.mplug)
@MODELS.register_module(Tasks.image_text_retrieval, module_name=Models.mplug)
class MPlugForAllTasks(TorchModel):

    def __init__(self, model_dir: str, task=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'initialize the mplug model from the `model_dir` path.\n        Args:\n            model_dir (str): the model path.\n        '
        super().__init__(model_dir, *args, **kwargs)
        from modelscope.models.multi_modal.mplug import MPlug
        self.model = MPlug.from_pretrained(model_dir, task=task)
        self.tokenizer = self.model.tokenizer

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if False:
            return 10
        "return the result by the model\n\n        Args:\n            input (Dict[str, Tensor]): the preprocessed data\n\n        Returns:\n            Dict[str, Tensor]: results\n                Example:\n                    {\n                        'predictions': Tensor([[1377, 4959, 2785, 6392...])]),\n                    }\n        "
        task = Config.from_file(osp.join(self.model_dir, ModelFile.CONFIGURATION)).task
        if not self.training and 'question' in input:
            output = self.model(input['image'], input['question'], train=False)
            if task == Tasks.image_text_retrieval:
                return {OutputKeys.SCORES: output[0].tolist()}
            (topk_ids, _) = output
            pred_string: List[str] = self.tokenizer.decode(topk_ids[0][0], skip_special_tokens=True)
            output_key = OutputKeys.CAPTION if task == Tasks.image_captioning else OutputKeys.TEXT
            return {output_key: pred_string}
        import addict
        image = input['image']
        answer = addict.Dict(input_ids=input['answer_input_ids'], attention_mask=input['answer_attention_mask'])
        if 'index' not in input:
            question = addict.Dict(input_ids=input['question_input_ids'], attention_mask=input['question_attention_mask'])
            output = self.model(image, question, answer, train=self.training)
        else:
            index = input['index']
            output = self.model(image, answer, index, train=self.training)
        if self.training:
            return {OutputKeys.LOSS: output}
        (topk_ids, _) = output
        return {'sequences': [list_tensor[0] for list_tensor in topk_ids]}

@MODELS.register_module(Tasks.video_question_answering, module_name=Models.hitea)
@MODELS.register_module(Tasks.video_captioning, module_name=Models.hitea)
class HiTeAForAllTasks(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        if False:
            while True:
                i = 10
        'initialize the hitea model from the `model_dir` path.\n        Args:\n            model_dir (str): the model path.\n        '
        super().__init__(model_dir, *args, **kwargs)
        from modelscope.models.multi_modal.mplug import HiTeA
        self.model = HiTeA.from_pretrained(model_dir)
        self.tokenizer = self.model.tokenizer

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if False:
            i = 10
            return i + 15
        "return the result by the model\n\n        Args:\n            input (Dict[str, Tensor]): the preprocessed data\n\n        Returns:\n            Dict[str, Tensor]: results\n                Example:\n                    {\n                        'predictions': Tensor([[1377, 4959, 2785, 6392...])]),\n                    }\n        "
        task = Config.from_file(osp.join(self.model_dir, ModelFile.CONFIGURATION)).task
        if not self.training and 'question' in input:
            output = self.model(input['video'], input['question'], train=False)
            (topk_ids, _) = output
            pred_string: List[str] = self.tokenizer.decode(topk_ids[0][0], skip_special_tokens=True)
            output_key = OutputKeys.CAPTION if task == Tasks.video_captioning else OutputKeys.TEXT
            return {output_key: pred_string}
        import addict
        video = input['video']
        answer = addict.Dict(input_ids=input['answer_input_ids'], attention_mask=input['answer_attention_mask'])
        if 'index' not in input:
            question = addict.Dict(input_ids=input['question_input_ids'], attention_mask=input['question_attention_mask'])
            output = self.model(video, question, answer, train=self.training)
        else:
            index = input['index']
            output = self.model(video, answer, index, train=self.training)
        if self.training:
            return {OutputKeys.LOSS: output}
        (topk_ids, _) = output
        return {'sequences': [list_tensor[0] for list_tensor in topk_ids]}
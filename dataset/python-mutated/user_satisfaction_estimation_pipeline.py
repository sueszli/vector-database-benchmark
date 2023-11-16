from typing import Any, Dict, Union
import numpy as np
import torch
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import DialogueClassificationUsePreprocessor
from modelscope.utils.constant import Tasks
__all__ = ['UserSatisfactionEstimationPipeline']

@PIPELINES.register_module(group_key=Tasks.text_classification, module_name=Pipelines.user_satisfaction_estimation)
class UserSatisfactionEstimationPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], preprocessor: DialogueClassificationUsePreprocessor=None, config_file: str=None, device: str='gpu', auto_collate=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "The inference pipeline for the user satisfaction estimation task.\n\n        Args:\n            model (str or Model): Supply either a local model dir which supported user satisfaction estimation task, or\n            a model id from the model hub, or a torch model instance.\n            preprocessor (DialogueClassificationUsePreprocessor): An optional preprocessor instance.\n            device (str): device str, should be either cpu, cuda, gpu, gpu:X or cuda:X\n            auto_collate (bool): automatically to convert data to tensor or not.\n\n        Examples:\n            >>> from modelscope.pipelines import pipeline\n            >>> pipeline_ins = pipeline('text-classification',\n                model='damo/nlp_user-satisfaction-estimation_chinese')\n            >>> input = [('返修退换货咨询|||', '手机有质量问题怎么办|||稍等，我看下', '开不开机了|||',\n                       '说话|||谢谢哈')]\n            >>> print(pipeline_ins(input))\n        "
        super().__init__(model=model, preprocessor=preprocessor, config_file=config_file, device=device, auto_collate=auto_collate, compile=kwargs.pop('compile', False), compile_options=kwargs.pop('compile_options', {}))
        if hasattr(self.preprocessor, 'id2label'):
            self.id2label = self.preprocessor.id2label
        self.model.eval()

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        with torch.no_grad():
            return self.model(**inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any], topk: int=None) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Process the prediction results\n\n                Args:\n                    inputs (`Dict[str, Any]` or `DialogueUseClassificationModelOutput`): The model output, please check\n                        the `DialogueUseClassificationModelOutput` class for details.\n                    topk (int): The topk probs to take\n                Returns:\n                    Dict[str, Any]: the prediction results.\n                        scores: The probabilities of each label.\n                        labels: The real labels.\n                    Label at index 0 is the largest probability.\n                '
        logits = inputs[OutputKeys.LOGITS].cpu().numpy()
        if logits.shape[0] == 1:
            logits = logits[0]

        def softmax(logits):
            if False:
                return 10
            exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            return exp / exp.sum(axis=-1, keepdims=True)
        probs = softmax(logits)
        num_classes = probs.shape[-1]
        topk = min(topk, num_classes) if topk is not None else num_classes
        top_indices = np.argpartition(probs, -topk)[-topk:]
        probs = np.take_along_axis(probs, top_indices, axis=-1).tolist()

        def map_to_label(_id):
            if False:
                print('Hello World!')
            if getattr(self, 'id2label', None) is not None:
                if _id in self.id2label:
                    return self.id2label[_id]
                elif str(_id) in self.id2label:
                    return self.id2label[str(_id)]
                else:
                    raise Exception(f'id {_id} not found in id2label: {self.id2label}')
            else:
                return _id
        v_func = np.vectorize(map_to_label)
        top_indices = v_func(top_indices).tolist()
        probs = list(reversed(probs))
        top_indices = list(reversed(top_indices))
        return {OutputKeys.SCORES: probs, OutputKeys.LABELS: top_indices}
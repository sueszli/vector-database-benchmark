from typing import Any, Dict, Union
import torch
from scipy.special import softmax
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModelFile, Tasks
__all__ = ['ZeroShotClassificationPipeline']

@PIPELINES.register_module(Tasks.zero_shot_classification, module_name=Pipelines.zero_shot_classification)
class ZeroShotClassificationPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], preprocessor: Preprocessor=None, config_file: str=None, device: str='gpu', auto_collate=True, sequence_length=512, **kwargs):
        if False:
            i = 10
            return i + 15
        "Use `model` and `preprocessor` to create a nlp zero shot classifiction for prediction.\n\n        A zero-shot classification task is used to classify texts by prompts.\n        In a normal classification task, model may produce a positive label by the input text\n        like 'The ice cream is made of the high quality milk, it is so delicious'\n        In a zero-shot task, the sentence is converted to:\n        ['The ice cream is made of the high quality milk, it is so delicious', 'This means it is good']\n        And:\n        ['The ice cream is made of the high quality milk, it is so delicious', 'This means it is bad']\n        Then feed these sentences into the model and turn the task to a NLI task(entailment, contradiction),\n        and compare the output logits to give the original classification label.\n\n\n        Args:\n            model (str or Model): Supply either a local model dir which supported the task,\n            or a model id from the model hub, or a torch model instance.\n            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for\n            the model if supplied.\n            kwargs (dict, `optional`):\n                Extra kwargs passed into the preprocessor's constructor.\n\n        Examples:\n            >>> from modelscope.pipelines import pipeline\n            >>> pipeline_ins = pipeline(task='zero-shot-classification',\n            >>>    model='damo/nlp_structbert_zero-shot-classification_chinese-base')\n            >>> sentence1 = '全新突破 解放军运20版空中加油机曝光'\n            >>> labels = ['文化', '体育', '娱乐', '财经', '家居', '汽车', '教育', '科技', '军事']\n            >>> template = '这篇文章的标题是{}'\n            >>> print(pipeline_ins(sentence1, candidate_labels=labels, hypothesis_template=template))\n\n            To view other examples plese check tests/pipelines/test_zero_shot_classification.py.\n        "
        super().__init__(model=model, preprocessor=preprocessor, config_file=config_file, device=device, auto_collate=auto_collate, compile=kwargs.pop('compile', False), compile_options=kwargs.pop('compile_options', {}))
        self.entailment_id = 0
        self.contradiction_id = 2
        assert isinstance(self.model, Model), f'please check whether model config exists in {ModelFile.CONFIGURATION}'
        if preprocessor is None:
            sequence_length = kwargs.pop('sequence_length', 512)
            self.preprocessor = Preprocessor.from_pretrained(self.model.model_dir, sequence_length=sequence_length, **kwargs)
        self.model.eval()

    def _sanitize_parameters(self, **kwargs):
        if False:
            print('Hello World!')
        preprocess_params = {}
        postprocess_params = {}
        if 'candidate_labels' in kwargs:
            candidate_labels = self._parse_labels(kwargs.pop('candidate_labels'))
            preprocess_params['candidate_labels'] = candidate_labels
            postprocess_params['candidate_labels'] = candidate_labels
        else:
            raise ValueError('You must include at least one label.')
        preprocess_params['hypothesis_template'] = kwargs.pop('hypothesis_template', '{}')
        postprocess_params['multi_label'] = kwargs.pop('multi_label', False)
        return (preprocess_params, {}, postprocess_params)

    def _parse_labels(self, labels):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(labels, str):
            labels = labels.replace('，', ',')
            labels = [label.strip() for label in labels.split(',') if label.strip()]
        return labels

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return self.model(**inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any], candidate_labels, multi_label=False) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'process the prediction results\n        Args:\n            inputs (Dict[str, Any]): _description_\n        Returns:\n            Dict[str, Any]: the prediction results\n        '
        logits = inputs[OutputKeys.LOGITS].cpu().numpy()
        if multi_label or len(candidate_labels) == 1:
            logits = logits[..., [self.contradiction_id, self.entailment_id]]
            scores = softmax(logits, axis=-1)[..., 1]
        else:
            logits = logits[..., self.entailment_id]
            scores = softmax(logits, axis=-1)
        reversed_index = list(reversed(scores.argsort()))
        result = {OutputKeys.LABELS: [candidate_labels[i] for i in reversed_index], OutputKeys.SCORES: [scores[i].item() for i in reversed_index]}
        return result
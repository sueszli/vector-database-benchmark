from typing import Any, Dict, Union
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import SpaceForDialogIntent
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import DialogIntentPredictionPreprocessor
from modelscope.utils.constant import Tasks
__all__ = ['DialogIntentPredictionPipeline']

@PIPELINES.register_module(Tasks.task_oriented_conversation, module_name=Pipelines.dialog_intent_prediction)
class DialogIntentPredictionPipeline(Pipeline):

    def __init__(self, model: Union[SpaceForDialogIntent, str], preprocessor: DialogIntentPredictionPreprocessor=None, config_file: str=None, device: str='gpu', auto_collate=True, **kwargs):
        if False:
            while True:
                i = 10
        "Use `model` and `preprocessor` to create a dialog intent prediction pipeline\n\n        Args:\n            model (str or SpaceForDialogIntent): Supply either a local model dir or a model id from the model hub,\n            or a SpaceForDialogIntent instance.\n            preprocessor (DialogIntentPredictionPreprocessor): An optional preprocessor instance.\n            kwargs (dict, `optional`):\n                Extra kwargs passed into the preprocessor's constructor.\n        "
        super().__init__(model=model, preprocessor=preprocessor, config_file=config_file, device=device, auto_collate=auto_collate, compile=kwargs.pop('compile', False), compile_options=kwargs.pop('compile_options', {}))
        if preprocessor is None:
            self.preprocessor = DialogIntentPredictionPreprocessor(self.model.model_dir, **kwargs)
        self.categories = self.preprocessor.categories

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        if False:
            print('Hello World!')
        'process the prediction results\n\n        Args:\n            inputs (Dict[str, Any]): _description_\n\n        Returns:\n            Dict[str, str]: the prediction results\n        '
        import numpy as np
        pred = inputs['pred']
        pos = np.where(pred == np.max(pred))
        return {OutputKeys.OUTPUT: {OutputKeys.PREDICTION: pred, OutputKeys.LABEL_POS: pos[0], OutputKeys.LABEL: self.categories[pos[0][0]]}}
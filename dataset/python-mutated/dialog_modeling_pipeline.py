from typing import Dict, Union
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import SpaceForDialogModeling
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import DialogModelingPreprocessor
from modelscope.utils.constant import Tasks
__all__ = ['DialogModelingPipeline']

@PIPELINES.register_module(Tasks.task_oriented_conversation, module_name=Pipelines.dialog_modeling)
class DialogModelingPipeline(Pipeline):

    def __init__(self, model: Union[SpaceForDialogModeling, str], preprocessor: DialogModelingPreprocessor=None, config_file: str=None, device: str='gpu', auto_collate=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Use `model` and `preprocessor` to create a dialog modeling pipeline for dialog response generation\n\n        Args:\n            model (str or SpaceForDialogModeling): Supply either a local model dir or a model id from the model hub,\n            or a SpaceForDialogModeling instance.\n            preprocessor (DialogModelingPreprocessor): An optional preprocessor instance.\n            kwargs (dict, `optional`):\n                Extra kwargs passed into the preprocessor's constructor.\n        "
        super().__init__(model=model, preprocessor=preprocessor, config_file=config_file, device=device, auto_collate=auto_collate)
        if preprocessor is None:
            self.preprocessor = DialogModelingPreprocessor(self.model.model_dir, **kwargs)

    def postprocess(self, inputs: Dict[str, Tensor]) -> Dict[str, str]:
        if False:
            while True:
                i = 10
        'process the prediction results\n\n        Args:\n            inputs (Dict[str, Any]): _description_\n\n        Returns:\n            Dict[str, str]: the prediction results\n        '
        sys_rsp = self.preprocessor.text_field.tokenizer.convert_ids_to_tokens(inputs['resp'])
        assert len(sys_rsp) > 2
        sys_rsp = sys_rsp[1:len(sys_rsp) - 1]
        inputs[OutputKeys.OUTPUT] = sys_rsp
        return inputs
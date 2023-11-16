from typing import Any, Dict, Union
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import SpaceForDST
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import DialogStateTrackingPreprocessor
from modelscope.utils.constant import Tasks
__all__ = ['DialogStateTrackingPipeline']

@PIPELINES.register_module(Tasks.task_oriented_conversation, module_name=Pipelines.dialog_state_tracking)
class DialogStateTrackingPipeline(Pipeline):

    def __init__(self, model: Union[SpaceForDST, str], preprocessor: DialogStateTrackingPreprocessor=None, config_file: str=None, device: str='gpu', auto_collate=True, **kwargs):
        if False:
            while True:
                i = 10
        "use `model` and `preprocessor` to create a dialog state tracking pipeline for\n        observation of dialog states tracking after many turns of open domain dialogue\n\n        Args:\n            model (str or SpaceForDialogStateTracking): Supply either a local model dir or a model id\n            from the model hub, or a SpaceForDialogStateTracking instance.\n            preprocessor (DialogStateTrackingPreprocessor): An optional preprocessor instance.\n            kwargs (dict, `optional`):\n                Extra kwargs passed into the preprocessor's constructor.\n        "
        super().__init__(model=model, preprocessor=preprocessor, config_file=config_file, device=device, auto_collate=auto_collate, compile=kwargs.pop('compile', False), compile_options=kwargs.pop('compile_options', {}))
        if preprocessor is None:
            self.preprocessor = DialogStateTrackingPreprocessor(self.model.model_dir, **kwargs)
        self.tokenizer = self.preprocessor.tokenizer
        self.config = self.preprocessor.config

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        if False:
            i = 10
            return i + 15
        'process the prediction results\n\n        Args:\n            inputs (Dict[str, Any]): _description_\n\n        Returns:\n            Dict[str, str]: the prediction results\n        '
        _inputs = inputs['inputs']
        _outputs = inputs['outputs']
        unique_ids = inputs['unique_ids']
        input_ids_unmasked = inputs['input_ids_unmasked']
        values = inputs['values']
        inform = inputs['inform']
        prefix = inputs['prefix']
        ds = inputs['ds']
        ds = predict_and_format(self.config, self.tokenizer, _inputs, _outputs[2], _outputs[3], _outputs[4], _outputs[5], unique_ids, input_ids_unmasked, values, inform, prefix, ds)
        return {OutputKeys.OUTPUT: ds}

def predict_and_format(config, tokenizer, features, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits, ids, input_ids_unmasked, values, inform, prefix, ds):
    if False:
        print('Hello World!')
    import re
    prediction_list = []
    dialog_state = ds
    for i in range(len(ids)):
        if int(ids[i].split('-')[2]) == 0:
            dialog_state = {slot: 'none' for slot in config.dst_slot_list}
        prediction = {}
        prediction_addendum = {}
        for slot in config.dst_slot_list:
            class_logits = per_slot_class_logits[slot][i]
            start_logits = per_slot_start_logits[slot][i]
            end_logits = per_slot_end_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]
            input_ids = features['input_ids'][i].tolist()
            class_label_id = int(features['class_label_id'][slot][i])
            start_pos = int(features['start_pos'][slot][i])
            end_pos = int(features['end_pos'][slot][i])
            refer_id = int(features['refer_id'][slot][i])
            class_prediction = int(class_logits.argmax())
            start_prediction = int(start_logits.argmax())
            end_prediction = int(end_logits.argmax())
            refer_prediction = int(refer_logits.argmax())
            prediction['guid'] = ids[i].split('-')
            prediction['class_prediction_%s' % slot] = class_prediction
            prediction['class_label_id_%s' % slot] = class_label_id
            prediction['start_prediction_%s' % slot] = start_prediction
            prediction['start_pos_%s' % slot] = start_pos
            prediction['end_prediction_%s' % slot] = end_prediction
            prediction['end_pos_%s' % slot] = end_pos
            prediction['refer_prediction_%s' % slot] = refer_prediction
            prediction['refer_id_%s' % slot] = refer_id
            prediction['input_ids_%s' % slot] = input_ids
            if class_prediction == config.dst_class_types.index('dontcare'):
                dialog_state[slot] = 'dontcare'
            elif class_prediction == config.dst_class_types.index('copy_value'):
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids_unmasked[i])
                dialog_state[slot] = ' '.join(input_tokens[start_prediction:end_prediction + 1])
                dialog_state[slot] = re.sub('(^| )##', '', dialog_state[slot])
            elif 'true' in config.dst_class_types and class_prediction == config.dst_class_types.index('true'):
                dialog_state[slot] = 'true'
            elif 'false' in config.dst_class_types and class_prediction == config.dst_class_types.index('false'):
                dialog_state[slot] = 'false'
            elif class_prediction == config.dst_class_types.index('inform'):
                if isinstance(inform[i][slot], str):
                    dialog_state[slot] = inform[i][slot]
                elif isinstance(inform[i][slot], list):
                    dialog_state[slot] = inform[i][slot][0]
            prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]
            prediction_addendum['slot_groundtruth_%s' % slot] = values[i][slot]
        for slot in config.dst_slot_list:
            class_logits = per_slot_class_logits[slot][i]
            refer_logits = per_slot_refer_logits[slot][i]
            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())
            if 'refer' in config.dst_class_types and class_prediction == config.dst_class_types.index('refer'):
                dialog_state[slot] = dialog_state[config.dst_slot_list[refer_prediction - 1]]
                prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]
        prediction.update(prediction_addendum)
        prediction_list.append(prediction)
    return dialog_state
from typing import Any, Dict
import torch
from modelscope.metainfo import Metrics, Trainers
from modelscope.outputs.outputs import ModelOutputBase
from modelscope.trainers import NlpEpochBasedTrainer
from modelscope.trainers.builder import TRAINERS

@TRAINERS.register_module(module_name=Trainers.text_generation_trainer)
class TextGenerationTrainer(NlpEpochBasedTrainer):

    def _decode(self, tokens):
        if False:
            return 10
        return self.eval_preprocessor.decode(tokens.tolist(), skip_special_tokens=True)

    def evaluation_step(self, data):
        if False:
            i = 10
            return i + 15
        model = self.model.module if self._dist else self.model
        model.eval()
        output = dict()
        with torch.no_grad():
            if Metrics.text_gen_metric in self.metrics:
                output.update(self._eval_genarate(model, data))
            if Metrics.PPL in self.metrics or Metrics.loss_metric in self.metrics:
                output.update(model.forward(**data))
        return output

    def _eval_genarate(self, model, data) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        result = model.generate(data)
        if isinstance(result, ModelOutputBase):
            result = result.to_dict()
        result['preds'] = [self._decode(seq) for seq in result['sequences']]
        data['tgts'] = [self._decode(seq) for seq in data['labels']]
        assert len(result['preds']) == len(data['tgts'])
        return result
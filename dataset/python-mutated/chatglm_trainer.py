from typing import Any, Dict, Union
import numpy as np
import torch
from transformers.deepspeed import is_deepspeed_zero3_enabled
from modelscope import EpochBasedTrainer, get_logger
logger = get_logger()

class Seq2SeqTrainer(EpochBasedTrainer):

    def _decode(self, tokens, ignore_pad_token_for_loss=False):
        if False:
            for i in range(10):
                print('nop')
        tokens = tokens.cpu().numpy()
        if ignore_pad_token_for_loss:
            tokens = np.where(tokens != -100, tokens, self.tokenizer.pad_token_id)
        tokens = np.where(tokens < self.tokenizer.vocab_size, tokens, self.tokenizer.pad_token_id)
        return [t for t in self.tokenizer.batch_decode(tokens, skip_special_tokens=True) if t != '</s>']

    def evaluation_step(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        if False:
            while True:
                i = 10
        has_labels = 'labels' in inputs
        gen_kwargs = self.cfg['gen_kwargs']
        if gen_kwargs.get('max_length') is None and gen_kwargs.get('max_new_tokens') is None:
            gen_kwargs['max_length'] = self.model.config.max_length
        gen_kwargs['num_beams'] = gen_kwargs['num_beams'] if gen_kwargs.get('num_beams') is not None else self.model.config.num_beams
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs['synced_gpus'] = gen_kwargs['synced_gpus'] if gen_kwargs.get('synced_gpus') is not None else default_synced_gpus
        if 'attention_mask' in inputs:
            gen_kwargs['attention_mask'] = inputs.get('attention_mask', None)
        if 'position_ids' in inputs:
            gen_kwargs['position_ids'] = inputs.get('position_ids', None)
        if 'global_attention_mask' in inputs:
            gen_kwargs['global_attention_mask'] = inputs.get('global_attention_mask', None)
        if hasattr(self.model, 'encoder') and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]
        gen_kwargs['input_ids'] = generation_inputs
        gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        self.model.eval()
        with torch.no_grad():
            generated_tokens = self.model.generate(**gen_kwargs)
        generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]
        if gen_kwargs.get('max_length') is not None and generated_tokens.shape[-1] < gen_kwargs['max_length']:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_length'])
        elif gen_kwargs.get('max_new_tokens') is not None and generated_tokens.shape[-1] < gen_kwargs['max_new_tokens'] + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_new_tokens'] + 1)
        if has_labels:
            labels = inputs['labels']
            if gen_kwargs.get('max_length') is not None and labels.shape[-1] < gen_kwargs['max_length']:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs['max_length'])
            elif gen_kwargs.get('max_new_tokens') is not None and labels.shape[-1] < gen_kwargs['max_new_tokens'] + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs['max_new_tokens'] + 1)
        else:
            labels = None
        generated_tokens = [''.join(self._decode(seq, False)) for seq in generated_tokens]
        inputs['tgts'] = [''.join(self._decode(seq, True)) for seq in labels]
        return {'preds': generated_tokens}

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if False:
            while True:
                i = 10
        if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id'):
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        elif self.model.config.pad_token_id is not None:
            pad_token_id = self.model.config.pad_token_id
        else:
            raise ValueError('Pad_token_id must be set in the configuration of the model, in order to pad tensors')
        padded_tensor = pad_token_id * torch.ones((tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:, :tensor.shape[-1]] = tensor
        return padded_tensor
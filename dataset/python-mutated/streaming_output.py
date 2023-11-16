import types
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Union
import torch
import torch.distributed as dist
from torch import nn
from transformers import PreTrainedModel
from transformers.generation import GreedySearchDecoderOnlyOutput
from transformers.generation import GreedySearchEncoderDecoderOutput, LogitsProcessorList, SampleDecoderOnlyOutput, SampleEncoderDecoderOutput, StoppingCriteriaList, validate_stopping_criteria
from modelscope.pipelines.base import Input
from modelscope.utils.constant import Frameworks
from modelscope.utils.device import device_placement

class StreamingOutputMixin:

    def stream_generate(self, *args, **kwargs) -> Generator:
        if False:
            i = 10
            return i + 15
        '\n        Support the input of Model and Pipeline.\n        The output is a `Generator` type,\n        which conforms to the output standard of modelscope.\n        '
        raise NotImplementedError

class PipelineStreamingOutputMixin(StreamingOutputMixin):

    def stream_generate(self, input: Union[Input, List[Input]], *args, **kwargs) -> Generator:
        if False:
            return 10
        '\n        Similar to the `Pipeline.__call__` method.\n        it supports the input that the pipeline can accept,\n        and also supports batch input.\n\n        self.model must be a subclass of StreamingOutputMixin\n        and implement the stream method.\n        '
        assert isinstance(self.model, StreamingOutputMixin), 'pipeline.model must be StreamingOutputMixin!'
        if self.model or (self.has_multiple_models and self.models[0]):
            if not self._model_prepare:
                self.prepare_model()
        batch_size = kwargs.pop('batch_size', None)
        (preprocess_params, forward_params, postprocess_params) = self._sanitize_parameters(**kwargs)
        if isinstance(input, list):
            model_input_list = [self._preprocess_with_check(i, preprocess_params) for i in input]
            if batch_size is None:
                output = []
                for ele in model_input_list:
                    output.append(self._stream_single(ele, forward_params, postprocess_params))
            else:
                output = self._stream_batch(model_input_list, batch_size, forward_params, postprocess_params)
        else:
            model_input = self._preprocess_with_check(input, preprocess_params)
            output = self._stream_single(model_input, forward_params, postprocess_params)
        return output

    def _preprocess_with_check(self, input: Input, preprocess_params: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        self._check_input(input)
        return self.preprocess(input, **preprocess_params)

    def _stream_single(self, model_input: Dict[str, Any], forward_params: Dict[str, Any], postprocess_params: Dict[str, Any]) -> Generator:
        if False:
            i = 10
            return i + 15
        with device_placement(self.framework, self.device_name):
            if self.framework == Frameworks.torch:
                with torch.no_grad():
                    if self._auto_collate:
                        model_input = self._collate_fn(model_input)
                    stream = self.model.stream_generate(model_input, **forward_params)
            else:
                stream = self.model.stream_generate(model_input, **forward_params)
            for out in stream:
                out = self.postprocess(out, **postprocess_params)
                self._check_output(out)
                yield out

    def _stream_batch(self, model_input_list: List[Dict[str, Any]], batch_size: int, forward_params: Dict[str, Any], postprocess_params: Dict[str, Any]) -> Generator:
        if False:
            for i in range(10):
                print('nop')
        stream_list = []
        real_batch_sizes = []
        with device_placement(self.framework, self.device_name):
            for i in range(0, len(model_input_list), batch_size):
                end = min(i + batch_size, len(model_input_list))
                real_batch_size = end - i
                real_batch_sizes.append(real_batch_size)
                batched_out = self._batch(model_input_list[i:end])
                if self.framework == Frameworks.torch:
                    with torch.no_grad():
                        if self._auto_collate:
                            batched_out = self._collate_fn(batched_out)
                        stream_list.append(self.model.stream_generate(batched_out, **forward_params))
                else:
                    stream_list.append(self.model.stream_generate(batched_out, **forward_params))
            output_list = [None] * len(model_input_list)
            stop_streams = 0
            while stop_streams < len(stream_list):
                stop_streams = 0
                for (i, (stream, real_batch_size)) in enumerate(zip(stream_list, real_batch_sizes)):
                    try:
                        batched_out = next(stream)
                        for batch_idx in range(real_batch_size):
                            out = {}
                            for (k, element) in batched_out.items():
                                if element is not None:
                                    if isinstance(element, (tuple, list)):
                                        if isinstance(element[0], torch.Tensor):
                                            out[k] = type(element)((e[batch_idx:batch_idx + 1] for e in element))
                                        else:
                                            out[k] = element[batch_idx]
                                    else:
                                        out[k] = element[batch_idx:batch_idx + 1]
                            out = self.postprocess(out, **postprocess_params)
                            self._check_output(out)
                            output_index = i * batch_size + batch_idx
                            output_list[output_index] = out
                    except StopIteration:
                        stop_streams += 1
                yield output_list
        return output_list

class PretrainedModelStreamingOutputMixin(StreamingOutputMixin):

    def stream_generate(self, *args, **kwargs) -> Generator:
        if False:
            for i in range(10):
                print('nop')
        model = self if isinstance(self, PreTrainedModel) else self.model
        assert isinstance(model, PreTrainedModel), 'self or self.model must be `PretrainedModel`!'
        with self._replace_generate(model):
            return model.generate(*args, **kwargs)

    @contextmanager
    def _replace_generate(self, model: PreTrainedModel) -> Generator:
        if False:
            i = 10
            return i + 15
        greedy_search = model.greedy_search
        sample = model.sample
        model.greedy_search = types.MethodType(self._greedy_search, model)
        model.sample = types.MethodType(self._sample, model)
        yield
        model.greedy_search = greedy_search
        model.sample = sample

    @staticmethod
    def _greedy_search(self, input_ids: torch.LongTensor, logits_processor: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, max_length: Optional[int]=None, pad_token_id: Optional[int]=None, eos_token_id: Optional[Union[int, List[int]]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_scores: Optional[bool]=None, return_dict_in_generate: Optional[bool]=None, synced_gpus: bool=False, **model_kwargs) -> Generator:
        if False:
            return 10
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn('`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.', UserWarning)
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        return_dict_in_generate = return_dict_in_generate if return_dict_in_generate is not None else self.generation_config.return_dict_in_generate
        scores = () if return_dict_in_generate and output_scores else None
        decoder_attentions = () if return_dict_in_generate and output_attentions else None
        cross_attentions = () if return_dict_in_generate and output_attentions else None
        decoder_hidden_states = () if return_dict_in_generate and output_hidden_states else None
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs['encoder_outputs'].get('attentions') if output_attentions else None
            encoder_hidden_states = model_kwargs['encoder_outputs'].get('hidden_states') if output_hidden_states else None
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        this_peer_finished = False
        while True:
            if synced_gpus:
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                if this_peer_finished_flag.item() == 0.0:
                    break
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs, return_dict=True, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            if synced_gpus and this_peer_finished:
                continue
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError('If `eos_token_id` is defined, make sure that `pad_token_id` is defined.')
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    yield GreedySearchEncoderDecoderOutput(sequences=input_ids, scores=scores, encoder_attentions=encoder_attentions, encoder_hidden_states=encoder_hidden_states, decoder_attentions=decoder_attentions, cross_attentions=cross_attentions, decoder_hidden_states=decoder_hidden_states)
                else:
                    yield GreedySearchDecoderOnlyOutput(sequences=input_ids, scores=scores, attentions=decoder_attentions, hidden_states=decoder_hidden_states)
            else:
                yield input_ids
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True
            if this_peer_finished and (not synced_gpus):
                break

    @staticmethod
    def _sample(self, input_ids: torch.LongTensor, logits_processor: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, logits_warper: Optional[LogitsProcessorList]=None, max_length: Optional[int]=None, pad_token_id: Optional[int]=None, eos_token_id: Optional[Union[int, List[int]]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_scores: Optional[bool]=None, return_dict_in_generate: Optional[bool]=None, synced_gpus: bool=False, **model_kwargs) -> Generator:
        if False:
            for i in range(10):
                print('nop')
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn('`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.', UserWarning)
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        return_dict_in_generate = return_dict_in_generate if return_dict_in_generate is not None else self.generation_config.return_dict_in_generate
        scores = () if return_dict_in_generate and output_scores else None
        decoder_attentions = () if return_dict_in_generate and output_attentions else None
        cross_attentions = () if return_dict_in_generate and output_attentions else None
        decoder_hidden_states = () if return_dict_in_generate and output_hidden_states else None
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs['encoder_outputs'].get('attentions') if output_attentions else None
            encoder_hidden_states = model_kwargs['encoder_outputs'].get('hidden_states') if output_hidden_states else None
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        this_peer_finished = False
        while True:
            if synced_gpus:
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                if this_peer_finished_flag.item() == 0.0:
                    break
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs, return_dict=True, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            if synced_gpus and this_peer_finished:
                continue
            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError('If `eos_token_id` is defined, make sure that `pad_token_id` is defined.')
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    yield SampleEncoderDecoderOutput(sequences=input_ids, scores=scores, encoder_attentions=encoder_attentions, encoder_hidden_states=encoder_hidden_states, decoder_attentions=decoder_attentions, cross_attentions=cross_attentions, decoder_hidden_states=decoder_hidden_states)
                else:
                    yield SampleDecoderOnlyOutput(sequences=input_ids, scores=scores, attentions=decoder_attentions, hidden_states=decoder_hidden_states)
            else:
                yield input_ids
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True
            if this_peer_finished and (not synced_gpus):
                break

def add_stream_generate(model: PreTrainedModel):
    if False:
        print('Hello World!')
    pretrained_class = type(model)
    parent_classes = (pretrained_class, PretrainedModelStreamingOutputMixin)
    new_model = type(pretrained_class.__name__, parent_classes, {})(model.config)
    new_model.__dict__.update(model.__dict__)
    return new_model
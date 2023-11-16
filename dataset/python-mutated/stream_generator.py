import copy
import inspect
import random
import warnings
from typing import Callable, List, Optional, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from transformers import BeamSearchScorer, ConstrainedBeamSearchScorer, DisjunctiveConstraint, GenerationConfig, GenerationMixin, LogitsProcessorList, PhrasalConstraint, PreTrainedModel, StoppingCriteriaList
from transformers.generation.utils import GenerateOutput, SampleOutput, logger

def setup_seed(seed):
    if False:
        i = 10
        return i + 15
    if seed == -1:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class StreamGenerationConfig(GenerationConfig):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.do_stream = kwargs.pop('do_stream', False)

class NewGenerationMixin(GenerationMixin):

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor]=None, generation_config: Optional[StreamGenerationConfig]=None, logits_processor: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]]=None, synced_gpus: Optional[bool]=False, seed=0, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        if False:
            i = 10
            return i + 15
        "\n\n        Generates sequences of token ids for models with a language modeling head.\n\n        <Tip warning={true}>\n\n        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the\n        model's default generation configuration. You can override any `generation_config` by passing the corresponding\n        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.\n\n        For an overview of generation strategies and code examples, check out the [following\n        guide](./generation_strategies).\n\n        </Tip>\n\n        Parameters:\n            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):\n                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the\n                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`\n                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of\n                `input_ids`, `input_values`, `input_features`, or `pixel_values`.\n            generation_config (`~generation.GenerationConfig`, *optional*):\n                The generation configuration to be used as base parametrization for the generation call. `**kwargs`\n                passed to generate matching the attributes of `generation_config` will override them. If\n                `generation_config` is not provided, the default will be used, which had the following loading\n                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model\n                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s\n                default values, whose documentation should be checked to parameterize generation.\n            logits_processor (`LogitsProcessorList`, *optional*):\n                Custom logits processors that complement the default logits processors built from arguments and\n                generation config. If a logit processor is passed that is already created with the arguments or a\n                generation config an error is thrown. This feature is intended for advanced users.\n            stopping_criteria (`StoppingCriteriaList`, *optional*):\n                Custom stopping criteria that complement the default stopping criteria built from arguments and a\n                generation config. If a stopping criteria is passed that is already created with the arguments or a\n                generation config an error is thrown. This feature is intended for advanced users.\n            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):\n                If provided, this function constraints the beam search to allowed tokens only at each step. If not\n                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and\n                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned\n                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful\n                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity\n                Retrieval](https://arxiv.org/abs/2010.00904).\n            synced_gpus (`bool`, *optional*, defaults to `False`):\n                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)\n            kwargs:\n                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be\n                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder\n                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.\n\n        Return:\n            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`\n            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.\n\n                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible\n                [`~utils.ModelOutput`] types are:\n\n                    - [`~generation.GreedySearchDecoderOnlyOutput`],\n                    - [`~generation.SampleDecoderOnlyOutput`],\n                    - [`~generation.BeamSearchDecoderOnlyOutput`],\n                    - [`~generation.BeamSampleDecoderOnlyOutput`]\n\n                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible\n                [`~utils.ModelOutput`] types are:\n\n                    - [`~generation.GreedySearchEncoderDecoderOutput`],\n                    - [`~generation.SampleEncoderDecoderOutput`],\n                    - [`~generation.BeamSearchEncoderDecoderOutput`],\n                    - [`~generation.BeamSampleEncoderDecoderOutput`]\n        "
        self._validate_model_class()
        if generation_config is None:
            if self.generation_config._from_model_config:
                new_generation_config = StreamGenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn('You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. Please use a generation configuration file (see https://huggingface.co/docs/transformers/main_classes/text_generation)')
                    self.generation_config = new_generation_config
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get('attention_mask', None) is None:
                logger.warning("The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f'Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.')
            generation_config.pad_token_id = eos_token_id
        (inputs_tensor, model_input_name, model_kwargs) = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        model_kwargs['output_attentions'] = generation_config.output_attentions
        model_kwargs['output_hidden_states'] = generation_config.output_hidden_states
        model_kwargs['use_cache'] = generation_config.use_cache
        accepts_attention_mask = 'attention_mask' in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = 'encoder_outputs' not in model_kwargs
        if model_kwargs.get('attention_mask', None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs['attention_mask'] = self._prepare_attention_mask_for_generation(inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id)
        if not self.config.is_encoder_decoder:
            if generation_config.pad_token_id is not None and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0:
                logger.warning("A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.")
        if self.config.is_encoder_decoder and 'encoder_outputs' not in model_kwargs:
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name)
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(batch_size, decoder_start_token_id=generation_config.decoder_start_token_id, bos_token_id=generation_config.bos_token_id, model_kwargs=model_kwargs, device=inputs_tensor.device)
        else:
            input_ids = inputs_tensor
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get('max_length') is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(f'Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to {generation_config.max_length} (`generation_config.max_length`). Controlling `max_length` via the config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.', UserWarning)
        elif has_default_max_length and generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        elif not has_default_max_length and generation_config.max_new_tokens is not None:
            raise ValueError('Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a limit to the generated output length. Remove one of those arguments. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)')
        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(f'Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than the maximum length ({generation_config.max_length})')
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = 'decoder_input_ids' if self.config.is_encoder_decoder else 'input_ids'
            logger.warning(f'Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to {generation_config.max_length}. This can lead to unexpected behavior. You should consider increasing `max_new_tokens`.')
        is_constraint_gen_mode = generation_config.constraints is not None or generation_config.force_words_ids is not None
        is_contrastive_search_gen_mode = generation_config.top_k is not None and generation_config.top_k > 1 and (generation_config.do_sample is False) and (generation_config.penalty_alpha is not None) and (generation_config.penalty_alpha > 0)
        is_greedy_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is False) and (not is_constraint_gen_mode) and (not is_contrastive_search_gen_mode)
        is_sample_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is True) and (generation_config.do_stream is False) and (not is_constraint_gen_mode) and (not is_contrastive_search_gen_mode)
        is_sample_gen_stream_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_stream is True) and (not is_constraint_gen_mode) and (not is_contrastive_search_gen_mode)
        is_beam_gen_mode = generation_config.num_beams > 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is False) and (not is_constraint_gen_mode) and (not is_contrastive_search_gen_mode)
        is_beam_sample_gen_mode = generation_config.num_beams > 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is True) and (not is_constraint_gen_mode) and (not is_contrastive_search_gen_mode)
        is_group_beam_gen_mode = generation_config.num_beams > 1 and generation_config.num_beam_groups > 1 and (not is_constraint_gen_mode) and (not is_contrastive_search_gen_mode)
        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError('`num_beam_groups` has to be smaller or equal to `num_beams`')
        if is_group_beam_gen_mode and generation_config.do_sample is True:
            raise ValueError('Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`.')
        if self.device.type != input_ids.device.type:
            warnings.warn(f"You are calling .generate() with the `input_ids` being on a device type different than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model is on {self.device.type}. You may experience unexpected behaviors or slower generation. Please make sure that you have put `input_ids` to the correct device by calling for example input_ids = input_ids.to('{self.device.type}') before running `.generate()`.", UserWarning)
        logits_processor = self._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_seq_length, encoder_input_ids=inputs_tensor, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, logits_processor=logits_processor)
        stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria)
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(f'num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing greedy search.')
            return self.greedy_search(input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, **model_kwargs)
        elif is_contrastive_search_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(f'num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing contrastive search.')
            return self.contrastive_search(input_ids, top_k=generation_config.top_k, penalty_alpha=generation_config.penalty_alpha, logits_processor=logits_processor, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, **model_kwargs)
        elif is_sample_gen_mode:
            logits_warper = self._get_logits_warper(generation_config)
            (input_ids, model_kwargs) = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_return_sequences, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
            return self.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, **model_kwargs)
        elif is_sample_gen_stream_mode:
            logits_warper = self._get_logits_warper(generation_config)
            (input_ids, model_kwargs) = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_return_sequences, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
            return self.sample_stream(input_ids, logits_processor=logits_processor, logits_warper=logits_warper, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, **model_kwargs)
        elif is_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError('`num_return_sequences` has to be smaller or equal to `num_beams`.')
            if stopping_criteria.max_length is None:
                raise ValueError('`max_length` needs to be a stopping_criteria for now.')
            beam_scorer = BeamSearchScorer(batch_size=batch_size, num_beams=generation_config.num_beams, device=inputs_tensor.device, length_penalty=generation_config.length_penalty, do_early_stopping=generation_config.early_stopping, num_beam_hyps_to_keep=generation_config.num_return_sequences)
            (input_ids, model_kwargs) = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
            return self.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, **model_kwargs)
        elif is_beam_sample_gen_mode:
            logits_warper = self._get_logits_warper(generation_config)
            if stopping_criteria.max_length is None:
                raise ValueError('`max_length` needs to be a stopping_criteria for now.')
            beam_scorer = BeamSearchScorer(batch_size=batch_size * generation_config.num_return_sequences, num_beams=generation_config.num_beams, device=inputs_tensor.device, length_penalty=generation_config.length_penalty, do_early_stopping=generation_config.early_stopping)
            (input_ids, model_kwargs) = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_beams * generation_config.num_return_sequences, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
            return self.beam_sample(input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, **model_kwargs)
        elif is_group_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError('`num_return_sequences` has to be smaller or equal to `num_beams`.')
            if generation_config.num_beams % generation_config.num_beam_groups != 0:
                raise ValueError('`num_beams` should be divisible by `num_beam_groups` for group beam search.')
            if stopping_criteria.max_length is None:
                raise ValueError('`max_length` needs to be a stopping_criteria for now.')
            has_default_typical_p = kwargs.get('typical_p') is None and generation_config.typical_p == 1.0
            if not has_default_typical_p:
                raise ValueError('Decoder argument `typical_p` is not supported with beam groups.')
            beam_scorer = BeamSearchScorer(batch_size=batch_size, num_beams=generation_config.num_beams, max_length=stopping_criteria.max_length, device=inputs_tensor.device, length_penalty=generation_config.length_penalty, do_early_stopping=generation_config.early_stopping, num_beam_hyps_to_keep=generation_config.num_return_sequences, num_beam_groups=generation_config.num_beam_groups)
            (input_ids, model_kwargs) = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
            return self.group_beam_search(input_ids, beam_scorer, logits_processor=logits_processor, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, **model_kwargs)
        elif is_constraint_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError('`num_return_sequences` has to be smaller or equal to `num_beams`.')
            if stopping_criteria.max_length is None:
                raise ValueError('`max_length` needs to be a stopping_criteria for now.')
            if generation_config.num_beams <= 1:
                raise ValueError('`num_beams` needs to be greater than 1 for constrained generation.')
            if generation_config.do_sample:
                raise ValueError('`do_sample` needs to be false for constrained generation.')
            if generation_config.num_beam_groups is not None and generation_config.num_beam_groups > 1:
                raise ValueError('`num_beam_groups` not supported yet for constrained generation.')
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints
            if generation_config.force_words_ids is not None:

                def typeerror():
                    if False:
                        print('Hello World!')
                    raise ValueError(f'`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`of positive integers, but is {generation_config.force_words_ids}.')
                if not isinstance(generation_config.force_words_ids, list) or len(generation_config.force_words_ids) == 0:
                    typeerror()
                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_ids, list) for token_ids in word_ids)):
                            typeerror()
                        if any((any((not isinstance(token_id, int) or token_id < 0 for token_id in token_ids)) for token_ids in word_ids)):
                            typeerror()
                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0 for token_id in word_ids)):
                            typeerror()
                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)
            constrained_beam_scorer = ConstrainedBeamSearchScorer(constraints=final_constraints, batch_size=batch_size, num_beams=generation_config.num_beams, device=inputs_tensor.device, length_penalty=generation_config.length_penalty, do_early_stopping=generation_config.early_stopping, num_beam_hyps_to_keep=generation_config.num_return_sequences)
            (input_ids, model_kwargs) = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
            return self.constrained_beam_search(input_ids, constrained_beam_scorer=constrained_beam_scorer, logits_processor=logits_processor, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, **model_kwargs)

    @torch.no_grad()
    def sample_stream(self, input_ids: torch.LongTensor, logits_processor: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, logits_warper: Optional[LogitsProcessorList]=None, max_length: Optional[int]=None, pad_token_id: Optional[int]=None, eos_token_id: Optional[Union[int, List[int]]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_scores: Optional[bool]=None, return_dict_in_generate: Optional[bool]=None, synced_gpus: Optional[bool]=False, **model_kwargs) -> Union[SampleOutput, torch.LongTensor]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and\n        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.\n\n        <Tip warning={true}>\n\n        In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.\n        For an overview of generation strategies and code examples, check the [following\n        guide](./generation_strategies).\n\n        </Tip>\n\n        Parameters:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                The sequence used as a prompt for the generation.\n            logits_processor (`LogitsProcessorList`, *optional*):\n                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]\n                used to modify the prediction scores of the language modeling head applied at each generation step.\n            stopping_criteria (`StoppingCriteriaList`, *optional*):\n                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]\n                used to tell if the generation loop should stop.\n            logits_warper (`LogitsProcessorList`, *optional*):\n                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used\n                to warp the prediction score distribution of the language modeling head applied before multinomial\n                sampling at each generation step.\n            max_length (`int`, *optional*, defaults to 20):\n                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated\n                tokens. The maximum length of the sequence to be generated.\n            pad_token_id (`int`, *optional*):\n                The id of the *padding* token.\n            eos_token_id (`int`, *optional*):\n                The id of the *end-of-sequence* token.\n            output_attentions (`bool`, *optional*, defaults to `False`):\n                Whether or not to return the attentions tensors of all attention layers. See `attentions` under\n                returned tensors for more details.\n            output_hidden_states (`bool`, *optional*, defaults to `False`):\n                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors\n                for more details.\n            output_scores (`bool`, *optional*, defaults to `False`):\n                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.\n            return_dict_in_generate (`bool`, *optional*, defaults to `False`):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n            synced_gpus (`bool`, *optional*, defaults to `False`):\n                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)\n            model_kwargs:\n                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is\n                an encoder-decoder model the kwargs should include `encoder_outputs`.\n\n        Return:\n            [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:\n            A `torch.LongTensor` containing the generated tokens (default behaviour) or a\n            [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and\n            `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if\n            `model.config.is_encoder_decoder=True`.\n\n        Examples:\n\n        ```python\n        >>> from transformers import (\n        ...     AutoTokenizer,\n        ...     AutoModelForCausalLM,\n        ...     LogitsProcessorList,\n        ...     MinLengthLogitsProcessor,\n        ...     TopKLogitsWarper,\n        ...     TemperatureLogitsWarper,\n        ...     StoppingCriteriaList,\n        ...     MaxLengthCriteria,\n        ... )\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")\n        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")\n\n        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token\n        >>> model.config.pad_token_id = model.config.eos_token_id\n        >>> model.generation_config.pad_token_id = model.config.eos_token_id\n\n        >>> input_prompt = "Today is a beautiful day, and"\n        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids\n\n        >>> # instantiate logits processors\n        >>> logits_processor = LogitsProcessorList(\n        ...     [\n        ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),\n        ...     ]\n        ... )\n        >>> # instantiate logits processors\n        >>> logits_warper = LogitsProcessorList(\n        ...     [\n        ...         TopKLogitsWarper(50),\n        ...         TemperatureLogitsWarper(0.7),\n        ...     ]\n        ... )\n\n        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])\n\n        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT\n        >>> outputs = model.sample(\n        ...     input_ids,\n        ...     logits_processor=logits_processor,\n        ...     logits_warper=logits_warper,\n        ...     stopping_criteria=stopping_criteria,\n        ... )\n\n        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)\n        [\'Today is a beautiful day, and a wonderful day.\\n\\nI was lucky enough to meet the\']\n        ```'
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
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        return_dict_in_generate = return_dict_in_generate if return_dict_in_generate is not None else self.generation_config.return_dict_in_generate
        scores = () if return_dict_in_generate and output_scores else None
        decoder_attentions = () if return_dict_in_generate and output_attentions else None
        cross_attentions = () if return_dict_in_generate and output_attentions else None
        decoder_hidden_states = () if return_dict_in_generate and output_hidden_states else None
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
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
            yield (next_tokens, self.final_norm(outputs.hidden_states[-1][:, -1]))
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(sum((next_tokens != i for i in eos_token_id)).long())
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

def init_stream_support():
    if False:
        return 10
    'Overload PreTrainedModel for streaming.'
    PreTrainedModel.generate_stream = NewGenerationMixin.generate
    PreTrainedModel.sample_stream = NewGenerationMixin.sample_stream
if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
    PreTrainedModel.generate = NewGenerationMixin.generate
    PreTrainedModel.sample_stream = NewGenerationMixin.sample_stream
    model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-560m', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
    model = model.to('cuda:0')
    model = model.eval()
    prompt_text = 'hello? \n'
    input_ids = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids
    input_ids = input_ids.to('cuda:0')
    with torch.no_grad():
        result = model.generate(input_ids, max_new_tokens=200, do_sample=True, top_k=30, top_p=0.85, temperature=0.35, repetition_penalty=1.2, early_stopping=True, seed=0)
        print(tokenizer.decode(result, skip_special_tokens=True))
        generator = model.generate(input_ids, max_new_tokens=200, do_sample=True, top_k=30, top_p=0.85, temperature=0.35, repetition_penalty=1.2, early_stopping=True, seed=0, do_stream=True)
        stream_result = ''
        for x in generator:
            chunk = tokenizer.decode(x, skip_special_tokens=True)
            stream_result += chunk
        print(stream_result)
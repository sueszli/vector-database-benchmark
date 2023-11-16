import copy
import warnings
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionBackboneModelOutput, Seq2SeqLMOutput, TokenGeneratorOutput
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from .backbone import T5PreTrainedModel, T5Stack
from .configuration import T5Config
logger = get_logger()
__HEAD_MASK_WARNING_MSG = '\nThe input argument `head_mask` was split into two arguments `head_mask` and\n`decoder_head_mask`. Currently, `decoder_head_mask` is set to copy `head_mask`,\nbut this feature is deprecated and will be removed in future versions. If you do\nnot want to use any `decoder_head_mask` now, please set `decoder_head_mask =\ntorch.ones(num_layers, num_heads)`.\n'

@MODELS.register_module(group_key=Tasks.text2text_generation, module_name=Models.T5)
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = ['encoder\\.embed_tokens\\.weight', 'decoder\\.embed_tokens\\.weight', 'lm_head\\.weight']
    _keys_to_ignore_on_load_unexpected = ['decoder\\.block\\.0\\.layer\\.1\\.EncDecAttention\\.relative_attention_bias\\.weight']

    def __init__(self, config: T5Config, device_map=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()
        self.model_parallel = False
        if device_map == 'auto':
            self.parallelize()

    def parallelize(self, device_map=None):
        if False:
            i = 10
            return i + 15
        self.device_map = get_device_map(len(self.encoder.block), range(torch.cuda.device_count())) if device_map is None else device_map
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        if False:
            while True:
                i = 10
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to('cpu')
        self.decoder = self.decoder.to('cpu')
        self.lm_head = self.lm_head.to('cpu')
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        if False:
            return 10
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        if False:
            i = 10
            return i + 15
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        if False:
            for i in range(10):
                print('nop')
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        if False:
            while True:
                i = 10
        return self.lm_head

    def get_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.encoder

    def get_decoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.decoder

    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.FloatTensor]=None, decoder_head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        if False:
            return 10
        '\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary. T5 is a model\n                with relative position embeddings so you should be able to pad the\n                inputs on both the right and the left.\n\n                Indices can be obtained using [`T5Tokenizer`]. See\n                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`]\n                for detail.\n\n                [What are input IDs?](../glossary#input-ids)\n\n                To know more on how to prepare `input_ids` for pretraining take a\n                look a [T5 Training](./t5#training).\n            attention_mask (`torch.FloatTensor` of shape `(batch_size,sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask\n                values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n                Indices of decoder input sequence tokens in the vocabulary.\n\n                Indices can be obtained using [`T5Tokenizer`]. See\n                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`]\n                for details.\n\n                [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n                T5 uses the `pad_token_id` as the starting token for\n                `decoder_input_ids` generation. If `past_key_values` is used,\n                optionally only the last `decoder_input_ids` have to be input (see\n                `past_key_values`).\n\n                To know more on how to prepare `decoder_input_ids` for pretraining\n                take a look at [T5 Training](./t5#training).\n            decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n                Default behavior: generate a tensor that ignores pad tokens in\n                `decoder_input_ids`. Causal mask will also be used by default.\n            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n                Mask to nullify selected heads of the self-attention modules in the\n                encoder. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or\n                `(num_layers, num_heads)`, *optional*):\n                Mask to nullify selected heads of the self-attention modules in the\n                decoder. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):\n                    Mask to nullify selected heads of the cross-attention modules in\n                    the decoder. Mask values selected in `[0, 1]`:\n\n                    - 1 indicates the head is **not masked**,\n                    - 0 indicates the head is **masked**.\n\n            encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):\n                Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*,\n                `optional`: *attentions*) `last_hidden_state` of shape `(batch_size,\n                sequence_length, hidden_size)` is a sequence of hidden states at the\n                output of the last layer of the encoder. Used in the cross-attention\n                of the decoder.\n            past_key_values (`tuple(tuple(torch.FloatTensor))` of length\n                `config.n_layers` with each tuple having 4 tensors of shape\n                `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n\n                Contains precomputed key and value hidden states of the attention\n                blocks. Can be used to speed up decoding.\n\n                If `past_key_values` are used, the user can optionally input only\n                the last `decoder_input_ids` (those that don\'t have their past key\n                value states given to this model) of shape `(batch_size, 1)` instead\n                of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n                Optionally, instead of passing `input_ids` you can choose to\n                directly pass an embedded representation. This is useful if you want\n                more control over how to convert `input_ids` indices into associated\n                vectors than the model\'s internal embedding lookup matrix.\n            decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`,\n                *optional*):\n                Optionally, instead of passing `decoder_input_ids` you can choose to\n                directly pass an embedded representation. If `past_key_values` is\n                used, optionally only the last `decoder_inputs_embeds` have to be\n                input (see `past_key_values`). This is useful if you want more\n                control over how to convert `decoder_input_ids` indices into\n                associated vectors than the model\'s internal embedding lookup\n                matrix.\n\n                If `decoder_input_ids` and `decoder_inputs_embeds` are both unset,\n                `decoder_inputs_embeds` takes the value of `inputs_embeds`.\n\n            use_cache (`bool`, *optional*):\n                If set to `True`, `past_key_values` key value states are returned\n                and can be used to speed up decoding (see `past_key_values`).\n\n            output_attentions (`bool`, *optional*):\n                Whether or not to return the attentions tensors of all attention\n                layers. See `attentions` under returned tensors for more detail.\n            output_hidden_states (`bool`, *optional*):\n                Whether or not to return the hidden states of all layers. See\n                `hidden_states` under returned tensors for more detail.\n            return_dict (`bool`, *optional*):\n                Whether or not to return a [`~utils.ModelOutput`] instead of a plain\n                tuple.\n            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n                Labels for computing the sequence classification/regression loss.\n                Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All\n                labels set to `-100` are ignored (masked), the loss is only computed\n                for labels in `[0, ..., config.vocab_size]`\n\n        Returns:\n\n        Examples:\n\n        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration\n\n        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")\n        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")\n\n        >>> # training\n        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids\n        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids\n        >>> outputs = model(input_ids=input_ids, labels=labels)\n        >>> loss = outputs.loss\n        >>> logits = outputs.logits\n\n        >>> # inference\n        >>> input_ids = tokenizer(\n        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"\n        >>> ).input_ids  # Batch size 1\n        >>> outputs = model.generate(input_ids)\n        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))\n        >>> # studies have shown that owning a dog is good for you.\n        '
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, AttentionBackboneModelOutput)):
            encoder_outputs = AttentionBackboneModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        if labels is not None and decoder_input_ids is None and (decoder_inputs_embeds is None):
            decoder_input_ids = self._shift_right(labels)
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = decoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * self.model_dim ** (-0.5)
        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return (loss,) + output if loss is not None else output
        return Seq2SeqLMOutput(loss=loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            while True:
                i = 10
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {'decoder_input_ids': input_ids, 'past_key_values': past, 'encoder_outputs': encoder_outputs, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        if False:
            while True:
                i = 10
        return self._shift_right(labels)

    def generate(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        output = super().generate(*args, **kwargs)
        return TokenGeneratorOutput(sequences=output if isinstance(output, torch.Tensor) else output[0])

    def _reorder_cache(self, past, beam_idx):
        if False:
            while True:
                i = 10
        if past is None:
            logger.warning('You might want to consider setting `use_cache=True` to speed up decoding')
            return past
        reordered_decoder_past = ()
        for layer_past_states in past:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                reordered_layer_past_states = reordered_layer_past_states + (layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),)
            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
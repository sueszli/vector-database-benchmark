"""PyTorch Fairseq model, ported from https://github.com/pytorch/fairseq/tree/master/examples/wmt19"""
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, LayerNorm
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_end_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_fsmt import FSMTConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'facebook/wmt19-ru-en'
_CONFIG_FOR_DOC = 'FSMTConfig'
'\n\nHere is how to compare BLEU scores against fairseq implementation:\n\n# en-ru\n\nexport PAIR=en-ru\nexport DATA_DIR=data/$PAIR\nexport SAVE_DIR=data/$PAIR\nexport BS=8\nexport NUM_BEAMS=50\nmkdir -p $DATA_DIR\nsacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/val.source\nsacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/val.target\necho $PAIR\nPYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py facebook/wmt19-$PAIR $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS\n\n# (fairseq BLEU: 36.4 http://matrix.statmt.org/matrix/output/1914?score_id=37605)\n\n\n# ru-en\n\nexport PAIR=ru-en\nexport DATA_DIR=data/$PAIR\nexport SAVE_DIR=data/$PAIR\nexport BS=8\nexport NUM_BEAMS=50\nmkdir -p $DATA_DIR\nsacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/val.source\nsacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/val.target\nPYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py facebook/wmt19-$PAIR $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS\n\n\n# (fairseq BLEU: 41.3 http://matrix.statmt.org/matrix/output/1907?run_id=6937)\n\n\n# de-en\n\nexport PAIR=de-en\nexport DATA_DIR=data/$PAIR\nexport SAVE_DIR=data/$PAIR\nexport BS=8\nexport NUM_BEAMS=50\nmkdir -p $DATA_DIR\nsacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/val.source\nsacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/val.target\necho $PAIR\nPYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py facebook/wmt19-$PAIR $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS\n\n# (fairseq BLEU: 42.3 http://matrix.statmt.org/matrix/output/1902?run_id=6750)\n\n\n\n# en-de\n\nexport PAIR=en-de\nexport DATA_DIR=data/$PAIR\nexport SAVE_DIR=data/$PAIR\nexport BS=8\nmkdir -p $DATA_DIR\nsacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/val.source\nsacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/val.target\necho $PAIR\nPYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py facebook/wmt19-$PAIR $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS\n\n# (fairseq BLEU: 43.1 http://matrix.statmt.org/matrix/output/1909?run_id=6862)\n\n'
FSMT_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`FSMTConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n\n'
FSMT_GENERATION_EXAMPLE = '\n    Translation example::\n\n    ```python\n    >>> from transformers import AutoTokenizer, FSMTForConditionalGeneration\n\n    >>> mname = "facebook/wmt19-ru-en"\n    >>> model = FSMTForConditionalGeneration.from_pretrained(mname)\n    >>> tokenizer = AutoTokenizer.from_pretrained(mname)\n\n    >>> src_text = "Машинное обучение - это здорово, не так ли?"\n    >>> input_ids = tokenizer(src_text, return_tensors="pt").input_ids\n    >>> outputs = model.generate(input_ids, num_beams=5, num_return_sequences=3)\n    >>> tokenizer.decode(outputs[0], skip_special_tokens=True)\n    "Machine learning is great, isn\'t it?"\n    ```\n\n'
FSMT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`FSTMTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Indices of decoder input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are decoder input IDs?](../glossary#decoder-input-ids)\n\n            FSMT uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`\n            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).\n        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):\n            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0,\n            1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        encoder_outputs (`Tuple(torch.FloatTensor)`, *optional*):\n            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)\n            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden-states at\n            the output of the last layer of the encoder. Used in the cross-attention of the decoder.\n        past_key_values (`Tuple(torch.FloatTensor)` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):\n            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.\n            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that\n            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all\n            `decoder_input_ids` of shape `(batch_size, sequence_length)`.\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded\n            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be\n            input (see `past_key_values`). This is useful if you want more control over how to convert\n            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.\n\n            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value\n            of `inputs_embeds`.\n        use_cache (`bool`, *optional*, defaults to `True`):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

def invert_mask(attention_mask):
    if False:
        for i in range(10):
            print('nop')
    'Turns 1->0, 0->1, False->True, True-> False'
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)

def triu_onnx(x, diagonal=0):
    if False:
        for i in range(10):
            print('nop')
    l = x.shape[0]
    arange = torch.arange(l, device=x.device)
    mask = arange.expand(l, l)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    return x.masked_fill(mask == 0, 0)

def _prepare_fsmt_decoder_inputs(config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32):
    if False:
        return 10
    '\n    Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if none are provided.\n    This mimics the default behavior in fairseq. To override it pass in masks. Note: this is not called during\n    generation\n    '
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    (bsz, tgt_len) = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = triu_onnx(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len, dtype=causal_mask_dtype)), 1).to(device=decoder_input_ids.device)
    return (decoder_input_ids, decoder_padding_mask, causal_mask)

class PretrainedFSMTModel(PreTrainedModel):
    config_class = FSMTConfig
    base_model_prefix = 'model'

    def _init_weights(self, module):
        if False:
            print('Hello World!')
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        if False:
            i = 10
            return i + 15
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {'attention_mask': input_ids.ne(pad_token), 'input_ids': input_ids}
        return dummy_inputs

def _make_linear_from_emb(emb):
    if False:
        for i in range(10):
            print('nop')
    (vocab_size, emb_size) = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

def _check_shapes(shape_1, shape2):
    if False:
        print('Hello World!')
    if shape_1 != shape2:
        raise AssertionError(f'shape mismatch: {shape_1} != {shape2}')

def shift_tokens_right(input_ids, pad_token_id):
    if False:
        print('Hello World!')
    'Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).'
    input_ids.masked_fill_(input_ids == -100, pad_token_id)
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

def make_padding_mask(input_ids, padding_idx=1):
    if False:
        for i in range(10):
            print('nop')
    'True for pad tokens'
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask

class EncoderLayer(nn.Module):

    def __init__(self, config: FSMTConfig):
        if False:
            while True:
                i = 10
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, layer_head_mask, output_attentions=False):
        if False:
            return 10
        '\n        Args:\n            x (`torch.Tensor`): input to the layer of shape *(seq_len, batch, embed_dim)*\n            encoder_padding_mask (`torch.ByteTensor`): binary ByteTensor of shape\n                *(batch, src_len)* where padding elements are indicated by `1`.\n            for t_tgt, t_src is excluded (or masked out), =0 means it is\n            included in attention\n            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size\n                *(config.encoder_attention_heads,)*.\n\n        Returns:\n            encoded output of shape *(seq_len, batch, embed_dim)*\n        '
        residual = x
        (x, attn_weights) = self.self_attn(query=x, key=x, key_padding_mask=encoder_padding_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = nn.functional.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return (x, attn_weights)

class FSMTEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`EncoderLayer`].

    Args:
        config: FSMTConfig
    """

    def __init__(self, config: FSMTConfig, embed_tokens):
        if False:
            return 10
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.embed_tokens = embed_tokens
        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_positions = SinusoidalPositionalEmbedding(config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, inputs_embeds: torch.Tensor=None, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n            input_ids (`torch.LongTensor`): tokens in the source language of shape\n                *(batch, src_len)*\n            attention_mask (`torch.LongTensor`): indicating which indices are padding tokens\n            inputs_embeds (`torch.FloatTensor`):\n                embedding vectors of shape *(batch, src_len, embed_dim)*\n            head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n        Returns:\n            BaseModelOutput or Tuple comprised of:\n\n                - **x** (`torch.Tensor`): the last encoder layer's output of shape *(src_len, batch, embed_dim)*\n                - **encoder_states** (`Tuple(torch.FloatTensor`)): all intermediate hidden states of shape *(src_len,\n                  batch, embed_dim)*. Only populated if *output_hidden_states:* is True.\n                - **all_attentions** (`Tuple(torch.FloatTensor`)): Attention weights for each layer.\n                During training might not be of length n_layers because of layer dropout.\n        "
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            embed_pos = self.embed_positions(input_ids)
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds * self.embed_scale
            position_ids = inputs_embeds[:, :, 0].masked_fill(inputs_embeds[:, :, 0].eq(0), self.embed_positions.padding_idx)
            embed_pos = self.embed_positions(position_ids)
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        x = inputs_embeds + embed_pos
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers), f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.'
        for (idx, encoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                x = x.transpose(0, 1)
                encoder_states += (x,)
                x = x.transpose(0, 1)
            dropout_probability = torch.rand([])
            if self.training and dropout_probability < self.layerdrop:
                attn = None
            else:
                (x, attn) = encoder_layer(x, attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, output_attentions=output_attentions)
            if output_attentions:
                all_attentions = all_attentions + (attn,)
        x = x.transpose(0, 1)
        if output_hidden_states:
            encoder_states += (x,)
        if not return_dict:
            return tuple((v for v in [x, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)

class DecoderLayer(nn.Module):

    def __init__(self, config: FSMTConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = Attention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, encoder_decoder_attention=True)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_hidden_states, encoder_attn_mask=None, layer_state=None, causal_mask=None, layer_head_mask=None, cross_attn_layer_head_mask=None, decoder_padding_mask=None, output_attentions=False):
        if False:
            while True:
                i = 10
        residual = x
        if layer_state is None:
            layer_state = {}
        (x, self_attn_weights) = self.self_attn(query=x, key=x, layer_state=layer_state, key_padding_mask=decoder_padding_mask, attn_mask=causal_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        (x, cross_attn_weights) = self.encoder_attn(query=x, key=encoder_hidden_states, key_padding_mask=encoder_attn_mask, layer_state=layer_state, layer_head_mask=cross_attn_layer_head_mask, output_attentions=output_attentions)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = nn.functional.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return (x, self_attn_weights, layer_state, cross_attn_weights)

class FSMTDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DecoderLayer`]

    Args:
        config: FSMTConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: FSMTConfig, embed_tokens: nn.Embedding):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        embed_dim = embed_tokens.embedding_dim
        self.embed_positions = SinusoidalPositionalEmbedding(config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(self.embed_tokens.weight, modifier_rank=None):
                embed_tokens_weight_shape = self.embed_tokens.weight.shape
        else:
            embed_tokens_weight_shape = self.embed_tokens.weight.shape
        self.output_projection = nn.Linear(embed_tokens_weight_shape[1], embed_tokens_weight_shape[0], bias=False)
        self.output_projection.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, encoder_padding_mask: torch.Tensor, decoder_padding_mask: torch.Tensor, decoder_causal_mask: torch.Tensor, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, use_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if False:
            print('Hello World!')
        '\n        Includes several features from "Jointly Learning to Align and Translate with Transformer Models" (Garg et al.,\n        EMNLP 2019).\n\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch, tgt_len)`):\n                previous decoder outputs for teacher forcing\n            encoder_hidden_states: output from the encoder, used for\n                encoder-side attention\n            encoder_padding_mask: for ignoring pad tokens\n            past_key_values (dict or None): dictionary used for storing state during generation\n            head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):\n                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n            cross_attn_head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):\n                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:\n\n                - 1 indicates the head is **not masked**,\n                - 0 indicates the head is **masked**.\n\n        Returns:\n            BaseModelOutputWithPast or tuple:\n\n                - the decoder\'s features of shape *(batch, tgt_len, embed_dim)*\n                - the cache\n                - hidden states\n                - attentions\n        '
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            positions = self.embed_positions(input_ids)
            if use_cache:
                input_ids = input_ids[:, -1:]
                positions = positions[:, -1:]
            x = self.embed_tokens(input_ids) * self.embed_scale
        elif inputs_embeds is not None:
            position_ids = inputs_embeds[:, :, 0].masked_fill(inputs_embeds[:, :, 0].eq(0), self.embed_positions.padding_idx)
            positions = self.embed_positions(position_ids)
            x = inputs_embeds * self.embed_scale
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        x += positions
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None
        next_decoder_cache = []
        for (attn_mask, mask_name) in zip([head_mask, cross_attn_head_mask], ['head_mask', 'cross_attn_head_mask']):
            if attn_mask is not None:
                assert attn_mask.size()[0] == len(self.layers), f'The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.'
        for (idx, decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                x = x.transpose(0, 1)
                all_hidden_states += (x,)
                x = x.transpose(0, 1)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            layer_state = past_key_values[idx] if past_key_values is not None else None
            (x, layer_self_attn, layer_past, layer_cross_attn) = decoder_layer(x, encoder_hidden_states, encoder_attn_mask=encoder_padding_mask, decoder_padding_mask=decoder_padding_mask, layer_state=layer_state, causal_mask=decoder_causal_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, output_attentions=output_attentions)
            if use_cache:
                next_decoder_cache.append(layer_past.copy())
            if output_attentions:
                all_self_attns += (layer_self_attn,)
                all_cross_attns += (layer_cross_attn,)
        if output_hidden_states:
            x = x.transpose(0, 1)
            all_hidden_states += (x,)
            x = x.transpose(0, 1)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        x = self.output_projection(x)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple((v for v in [x, next_cache, all_hidden_states, all_self_attns, all_cross_attns] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attns)

def _reorder_buffer(attn_cache, new_order):
    if False:
        return 10
    for (k, input_buffer_k) in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache

class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, encoder_decoder_attention=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** (-0.5)
        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = 'encoder_decoder' if self.encoder_decoder_attention else 'self'

    def _shape(self, tensor, seq_len, bsz):
        if False:
            return 10
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(self, query, key: Optional[Tensor], key_padding_mask: Optional[Tensor]=None, layer_state: Optional[Dict[str, Optional[Tensor]]]=None, attn_mask: Optional[Tensor]=None, layer_head_mask: Optional[Tensor]=None, output_attentions=False) -> Tuple[Tensor, Optional[Tensor]]:
        if False:
            return 10
        'Input shape: Time(SeqLen) x Batch x Channel'
        static_kv: bool = self.encoder_decoder_attention
        (tgt_len, bsz, embed_dim) = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if layer_state is not None:
            saved_state = layer_state.get(self.cache_key, {})
            if 'prev_key' in saved_state and static_kv:
                key = None
        else:
            saved_state = None
            layer_state = {}
        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)
        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)
        if saved_state is not None:
            (k, v, key_padding_mask) = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)
        layer_state[self.cache_key] = {'prev_key': k.view(bsz, self.num_heads, -1, self.head_dim), 'prev_value': v.view(bsz, self.num_heads, -1, self.head_dim), 'prev_key_padding_mask': key_padding_mask if not static_kv else None}
        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)
        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, torch.finfo(attn_weights.dtype).min)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_heads,), f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}'
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped)

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        if False:
            print('Hello World!')
        if 'prev_key' in saved_state:
            _prev_key = saved_state['prev_key']
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if 'prev_value' in saved_state:
            _prev_value = saved_state['prev_value']
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get('prev_key_padding_mask', None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = key_padding_mask
        return (k, v, new_key_padding_mask)

def fill_with_neg_inf(t):
    if False:
        return 10
    'FP16-compatible function that fills a input_ids with -inf.'
    return t.float().fill_(torch.finfo(t.dtype).min).type_as(t)

def _get_shape(t):
    if False:
        print('Hello World!')
    return getattr(t, 'shape', None)

@add_start_docstrings('The bare FSMT Model outputting raw hidden-states without any specific head on top.', FSMT_START_DOCSTRING)
class FSMTModel(PretrainedFSMTModel):
    _tied_weights_keys = ['decoder.embed_tokens.weight', 'decoder.output_projection.weight']

    def __init__(self, config: FSMTConfig):
        if False:
            while True:
                i = 10
        super().__init__(config)
        padding_idx = config.pad_token_id
        encoder_embed_tokens = nn.Embedding(config.src_vocab_size, config.d_model, padding_idx)
        decoder_embed_tokens = nn.Embedding(config.tgt_vocab_size, config.d_model, padding_idx)
        self.encoder = FSMTEncoder(config, encoder_embed_tokens)
        self.decoder = FSMTDecoder(config, decoder_embed_tokens)
        self.post_init()

    def get_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.encoder

    def get_decoder(self):
        if False:
            return 10
        return self.decoder

    def _tie_weights(self):
        if False:
            i = 10
            return i + 15
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.get_input_embeddings())
            self._tie_or_clone_weights(self.decoder.output_projection, self.get_input_embeddings())

    @add_start_docstrings_to_model_forward(FSMT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[torch.FloatTensor]]=None, past_key_values: Optional[Tuple[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        if False:
            while True:
                i = 10
        if decoder_input_ids is None:
            use_cache = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not use_cache and input_ids is not None:
            (decoder_input_ids, decoder_padding_mask, causal_mask) = _prepare_fsmt_decoder_inputs(self.config, input_ids, decoder_input_ids=decoder_input_ids, decoder_padding_mask=decoder_attention_mask, causal_mask_dtype=self.decoder.embed_tokens.weight.dtype)
        else:
            (decoder_padding_mask, causal_mask) = (None, None)
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            raise ValueError('Make sure that `decoder_input_ids` or `decoder_inputs_embeds` are passed.')
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        decoder_outputs = self.decoder(decoder_input_ids, encoder_outputs[0], attention_mask, decoder_padding_mask, decoder_causal_mask=causal_mask, inputs_embeds=decoder_inputs_embeds, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return Seq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def get_input_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        if False:
            while True:
                i = 10
        self.encoder.embed_tokens = value

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.decoder.embed_tokens

    def set_output_embeddings(self, value):
        if False:
            print('Hello World!')
        self.decoder.embed_tokens = value

@add_start_docstrings('The FSMT Model with a language modeling head. Can be used for summarization.', FSMT_START_DOCSTRING)
class FSMTForConditionalGeneration(PretrainedFSMTModel):
    base_model_prefix = 'model'
    _tied_weights_keys = ['decoder.embed_tokens.weight', 'decoder.output_projection.weight']

    def __init__(self, config: FSMTConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        base_model = FSMTModel(config)
        self.model = base_model
        self.post_init()

    @add_start_docstrings_to_model_forward(FSMT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(FSMT_GENERATION_EXAMPLE)
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[torch.FloatTensor]]=None, past_key_values: Optional[Tuple[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.Tensor]=None, decoder_inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,\n            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored\n            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.\n\n        Returns:\n\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        outputs = self.model(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_inputs_embeds=decoder_inputs_embeds, encoder_outputs=encoder_outputs, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = outputs[0]
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.tgt_vocab_size), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        if False:
            print('Hello World!')
        return shift_tokens_right(labels, self.config.pad_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            while True:
                i = 10
        reordered_past = []
        for layer_past in past_key_values:
            layer_past_new = {attn_key: _reorder_buffer(attn_cache, beam_idx) for (attn_key, attn_cache) in layer_past.items()}
            reordered_past.append(layer_past_new)
        return reordered_past

    def get_encoder(self):
        if False:
            return 10
        return self.model.encoder

    def get_decoder(self):
        if False:
            print('Hello World!')
        return self.model.decoder

    def get_output_embeddings(self):
        if False:
            while True:
                i = 10
        return self.model.decoder.embed_tokens

    def set_output_embeddings(self, value):
        if False:
            while True:
                i = 10
        self.model.decoder.embed_tokens = value

class SinusoidalPositionalEmbedding(nn.Embedding):
    """
    This module produces sinusoidal positional embeddings of any length.

    We don't want to save the weight of this embedding since it's not trained (deterministic) and it can be huge.

    Padding symbols are ignored.

    These embeddings get automatically extended in forward if more positions is needed.
    """

    def __init__(self, num_positions, embedding_dim, padding_idx):
        if False:
            return 10
        self.make_weight(num_positions, embedding_dim, padding_idx)

    def make_weight(self, num_positions, embedding_dim, padding_idx):
        if False:
            while True:
                i = 10
        weight = self.get_embedding(num_positions, embedding_dim, padding_idx)
        if not hasattr(self, 'weight'):
            super().__init__(num_positions, embedding_dim, padding_idx, _weight=weight)
        else:
            weight = weight.to(dtype=self.weight.dtype, device=self.weight.device)
            self.weight = nn.Parameter(weight)
        self.weight.detach_()
        self.weight.requires_grad = False

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx):
        if False:
            print('Hello World!')
        '\n        Build sinusoidal embeddings.\n\n        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of\n        "Attention Is All You Need".\n        '
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    @staticmethod
    def make_positions(tensor, padding_idx: int):
        if False:
            return 10
        '\n        Replace non-padding symbols with their position numbers.\n\n        Position numbers begin at padding_idx+1. Padding symbols are ignored.\n        '
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

    def forward(self, input, incremental_state: Optional[Any]=None, timestep: Optional[Tensor]=None):
        if False:
            print('Hello World!')
        'Input is expected to be of size [bsz x seqlen].'
        (bsz, seq_len) = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weight.size(0):
            self.make_weight(max_pos, self.embedding_dim, self.padding_idx)
        positions = self.make_positions(input, self.padding_idx)
        return super().forward(positions)
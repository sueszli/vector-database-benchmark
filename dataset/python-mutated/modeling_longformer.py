"""PyTorch Longformer model."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_longformer import LongformerConfig
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'allenai/longformer-base-4096'
_CONFIG_FOR_DOC = 'LongformerConfig'
LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ['allenai/longformer-base-4096', 'allenai/longformer-large-4096', 'allenai/longformer-large-4096-finetuned-triviaqa', 'allenai/longformer-base-4096-extra.pos.embd.only', 'allenai/longformer-large-4096-extra.pos.embd.only']

@dataclass
class LongformerBaseModelOutput(ModelOutput):
    """
    Base class for Longformer's outputs, with potential hidden states, local and global attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LongformerBaseModelOutputWithPooling(ModelOutput):
    """
    Base class for Longformer's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LongformerMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LongformerQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering Longformer models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LongformerSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LongformerMultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice Longformer models.

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class LongformerTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None

def _get_question_end_index(input_ids, sep_token_id):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes the index of the first occurrence of `sep_token_id`.\n    '
    sep_token_indices = (input_ids == sep_token_id).nonzero()
    batch_size = input_ids.shape[0]
    assert sep_token_indices.shape[1] == 2, '`input_ids` should have two dimensions'
    assert sep_token_indices.shape[0] == 3 * batch_size, f'There should be exactly three separator tokens: {sep_token_id} in every sample for questions answering. You might also consider to set `global_attention_mask` manually in the forward function to avoid this error.'
    return sep_token_indices.view(batch_size, 3, 2)[:, 0, 1]

def _compute_global_attention_mask(input_ids, sep_token_id, before_sep_token=True):
    if False:
        i = 10
        return i + 15
    '\n    Computes global attention mask by putting attention on all tokens before `sep_token_id` if `before_sep_token is\n    True` else after `sep_token_id`.\n    '
    question_end_index = _get_question_end_index(input_ids, sep_token_id)
    question_end_index = question_end_index.unsqueeze(dim=1)
    attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
    if before_sep_token is True:
        attention_mask = (attention_mask.expand_as(input_ids) < question_end_index).to(torch.bool)
    else:
        attention_mask = (attention_mask.expand_as(input_ids) > question_end_index + 1).to(torch.bool) * (attention_mask.expand_as(input_ids) < input_ids.shape[-1]).to(torch.bool)
    return attention_mask

def create_position_ids_from_input_ids(input_ids, padding_idx):
    if False:
        return 10
    "\n    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols\n    are ignored. This is modified from fairseq's `utils.make_positions`.\n\n    Args:\n        x: torch.Tensor x:\n\n    Returns: torch.Tensor\n    "
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx

class LongformerEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if False:
            i = 10
            return i + 15
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        if False:
            print('Hello World!')
        '\n        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.\n\n        Args:\n            inputs_embeds: torch.Tensor inputs_embeds:\n\n        Returns: torch.Tensor\n        '
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape)

class LongformerSelfAttention(nn.Module):

    def __init__(self, config, layer_id):
        if False:
            print('Hello World!')
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.dropout = config.attention_probs_dropout_prob
        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert attention_window % 2 == 0, f'`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}'
        assert attention_window > 0, f'`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}'
        self.one_sided_attn_window_size = attention_window // 2
        self.config = config

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, is_index_masked=None, is_index_global_attn=None, is_global_attn=None, output_attentions=False):
        if False:
            print('Hello World!')
        '\n        [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to\n        *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.\n\n        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:\n\n            - -10000: no attention\n            - 0: local attention\n            - +10000: global attention\n        '
        hidden_states = hidden_states.transpose(0, 1)
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)
        (seq_len, batch_size, embed_dim) = hidden_states.size()
        assert embed_dim == self.embed_dim, f'hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}'
        query_vectors /= math.sqrt(self.head_dim)
        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        attn_scores = self._sliding_chunks_query_key_matmul(query_vectors, key_vectors, self.one_sided_attn_window_size)
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(remove_from_windowed_attention_mask, torch.finfo(query_vectors.dtype).min)
        diagonal_mask = self._sliding_chunks_query_key_matmul(float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size)
        attn_scores += diagonal_mask
        assert list(attn_scores.size()) == [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1], f'local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}'
        if is_global_attn:
            (max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero) = self._get_global_attn_indices(is_index_global_attn)
            global_key_attn_scores = self._concat_with_global_key_attn_probs(query_vectors=query_vectors, key_vectors=key_vectors, max_num_global_attn_indices=max_num_global_attn_indices, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)
            del global_key_attn_scores
        attn_probs = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_heads,), f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}'
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = attn_probs.type_as(attn_scores)
        del attn_scores
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        if is_global_attn:
            attn_output = self._compute_attn_output_with_global_indices(value_vectors=value_vectors, attn_probs=attn_probs, max_num_global_attn_indices=max_num_global_attn_indices, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero)
        else:
            attn_output = self._sliding_chunks_matmul_attn_probs_value(attn_probs, value_vectors, self.one_sided_attn_window_size)
        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), 'Unexpected size'
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        if is_global_attn:
            (global_attn_output, global_attn_probs) = self._compute_global_attn_output_from_hidden(hidden_states=hidden_states, max_num_global_attn_indices=max_num_global_attn_indices, layer_head_mask=layer_head_mask, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero, is_index_masked=is_index_masked)
            nonzero_global_attn_output = global_attn_output[is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]]
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(len(is_local_index_global_attn_nonzero[0]), -1)
            attn_probs[is_index_global_attn_nonzero] = 0
        outputs = (attn_output.transpose(0, 1),)
        if output_attentions:
            outputs += (attn_probs,)
        return outputs + (global_attn_probs,) if is_global_attn and output_attentions else outputs

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        if False:
            return 10
        'pads rows and then flips rows and columns'
        hidden_states_padded = nn.functional.pad(hidden_states_padded, padding)
        hidden_states_padded = hidden_states_padded.view(*hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2))
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        if False:
            return 10
        '\n        shift every row 1 step right, converting columns into diagonals.\n\n        Example:\n\n        ```python\n        chunked_hidden_states: [\n            0.4983,\n            2.6918,\n            -0.0071,\n            1.0492,\n            -1.8348,\n            0.7672,\n            0.2986,\n            0.0285,\n            -0.7584,\n            0.4206,\n            -0.0405,\n            0.1599,\n            2.0514,\n            -1.1600,\n            0.5372,\n            0.2629,\n        ]\n        window_overlap = num_rows = 4\n        ```\n\n                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000\n                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,\n                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]\n        '
        (total_num_heads, num_chunks, window_overlap, hidden_dim) = chunked_hidden_states.size()
        chunked_hidden_states = nn.functional.pad(chunked_hidden_states, (0, window_overlap + 1))
        chunked_hidden_states = chunked_hidden_states.view(total_num_heads, num_chunks, -1)
        chunked_hidden_states = chunked_hidden_states[:, :, :-window_overlap]
        chunked_hidden_states = chunked_hidden_states.view(total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim)
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap, onnx_export: bool=False):
        if False:
            i = 10
            return i + 15
        'convert into overlapping chunks. Chunk size = 2w, overlap size = w'
        if not onnx_export:
            hidden_states = hidden_states.view(hidden_states.size(0), torch.div(hidden_states.size(1), window_overlap * 2, rounding_mode='trunc'), window_overlap * 2, hidden_states.size(2))
            chunk_size = list(hidden_states.size())
            chunk_size[1] = chunk_size[1] * 2 - 1
            chunk_stride = list(hidden_states.stride())
            chunk_stride[1] = chunk_stride[1] // 2
            return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        chunk_size = [hidden_states.size(0), torch.div(hidden_states.size(1), window_overlap, rounding_mode='trunc') - 1, window_overlap * 2, hidden_states.size(2)]
        overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device)
        for chunk in range(chunk_size[1]):
            overlapping_chunks[:, chunk, :, :] = hidden_states[:, chunk * window_overlap:chunk * window_overlap + 2 * window_overlap, :]
        return overlapping_chunks

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        if False:
            print('Hello World!')
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, :affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :, :affected_seq_len + 1] = torch.full_like(beginning_input, -float('inf')).where(beginning_mask.bool(), beginning_input)
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):] = torch.full_like(ending_input, -float('inf')).where(ending_mask.bool(), ending_input)

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        if False:
            print('Hello World!')
        '\n        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This\n        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an\n        overlap of size window_overlap\n        '
        (batch_size, seq_len, num_heads, head_dim) = query.size()
        assert seq_len % (window_overlap * 2) == 0, f'Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}'
        assert query.size() == key.size()
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode='trunc') - 1
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        query = self._chunk(query, window_overlap, getattr(self.config, 'onnx_export', False))
        key = self._chunk(key, window_overlap, getattr(self.config, 'onnx_export', False))
        diagonal_chunked_attention_scores = torch.einsum('bcxd,bcyd->bcxy', (query, key))
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(diagonal_chunked_attention_scores, padding=(0, 0, 0, 1))
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros((batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1))
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[:, :, :window_overlap, :window_overlap + 1]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[:, -1, window_overlap:, :window_overlap + 1]
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[:, :, -(window_overlap + 1):-1, window_overlap + 1:]
        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[:, 0, :window_overlap - 1, 1 - window_overlap:]
        diagonal_attention_scores = diagonal_attention_scores.view(batch_size, num_heads, seq_len, 2 * window_overlap + 1).transpose(2, 1)
        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int):
        if False:
            i = 10
            return i + 15
        '\n        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the\n        same shape as `attn_probs`\n        '
        (batch_size, seq_len, num_heads, head_dim) = value.size()
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode='trunc') - 1
        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(batch_size * num_heads, torch.div(seq_len, window_overlap, rounding_mode='trunc'), window_overlap, 2 * window_overlap + 1)
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (chunked_value_stride[0], window_overlap * chunked_value_stride[1], chunked_value_stride[1], chunked_value_stride[2])
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        context = torch.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        if False:
            for i in range(10):
                print('nop')
        'compute global attn indices required throughout forward pass'
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)
        max_num_global_attn_indices = num_global_attn_indices.max()
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
        is_local_index_global_attn = torch.arange(max_num_global_attn_indices, device=is_index_global_attn.device) < num_global_attn_indices.unsqueeze(dim=-1)
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero)

    def _concat_with_global_key_attn_probs(self, key_vectors, query_vectors, max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero):
        if False:
            print('Hello World!')
        batch_size = key_vectors.shape[0]
        key_vectors_only_global = key_vectors.new_zeros(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim)
        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]
        attn_probs_from_global_key = torch.einsum('blhd,bshd->blhs', (query_vectors, key_vectors_only_global))
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(self, value_vectors, attn_probs, max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero):
        if False:
            for i in range(10):
                print('nop')
        batch_size = attn_probs.shape[0]
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        value_vectors_only_global = value_vectors.new_zeros(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim)
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]
        attn_output_only_global = torch.matmul(attn_probs_only_global.transpose(1, 2).clone(), value_vectors_only_global.transpose(1, 2).clone()).transpose(1, 2)
        attn_probs_without_global = attn_probs.narrow(-1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices).contiguous()
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(attn_probs_without_global, value_vectors, self.one_sided_attn_window_size)
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(self, hidden_states, max_num_global_attn_indices, layer_head_mask, is_local_index_global_attn_nonzero, is_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero, is_index_masked):
        if False:
            return 10
        (seq_len, batch_size) = hidden_states.shape[:2]
        global_attn_hidden_states = hidden_states.new_zeros(max_num_global_attn_indices, batch_size, self.embed_dim)
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = hidden_states[is_index_global_attn_nonzero[::-1]]
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)
        global_query_vectors_only_global /= math.sqrt(self.head_dim)
        global_query_vectors_only_global = global_query_vectors_only_global.contiguous().view(max_num_global_attn_indices, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_key_vectors = global_key_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_value_vectors = global_value_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_attn_scores = torch.bmm(global_query_vectors_only_global, global_key_vectors.transpose(1, 2))
        assert list(global_attn_scores.size()) == [batch_size * self.num_heads, max_num_global_attn_indices, seq_len], f'global_attn_scores have the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is {global_attn_scores.size()}.'
        global_attn_scores = global_attn_scores.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores[is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :] = torch.finfo(global_attn_scores.dtype).min
        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores = global_attn_scores.masked_fill(is_index_masked[:, None, None, :], torch.finfo(global_attn_scores.dtype).min)
        global_attn_scores = global_attn_scores.view(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_probs_float = nn.functional.softmax(global_attn_scores, dim=-1, dtype=torch.float32)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_heads,), f'Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}'
            global_attn_probs_float = layer_head_mask.view(1, -1, 1, 1) * global_attn_probs_float.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
            global_attn_probs_float = global_attn_probs_float.view(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_probs = nn.functional.dropout(global_attn_probs_float.type_as(global_attn_scores), p=self.dropout, training=self.training)
        global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)
        assert list(global_attn_output.size()) == [batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim], f'global_attn_output tensor has the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is {global_attn_output.size()}.'
        global_attn_probs = global_attn_probs.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_output = global_attn_output.view(batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim)
        return (global_attn_output, global_attn_probs)

class LongformerSelfOutput(nn.Module):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class LongformerAttention(nn.Module):

    def __init__(self, config, layer_id=0):
        if False:
            while True:
                i = 10
        super().__init__()
        self.self = LongformerSelfAttention(config, layer_id)
        self.output = LongformerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if False:
            while True:
                i = 10
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, is_index_masked=None, is_index_global_attn=None, is_global_attn=None, output_attentions=False):
        if False:
            return 10
        self_outputs = self.self(hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn, output_attentions=output_attentions)
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs

class LongformerIntermediate(nn.Module):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class LongformerOutput(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class LongformerLayer(nn.Module):

    def __init__(self, config, layer_id=0):
        if False:
            while True:
                i = 10
        super().__init__()
        self.attention = LongformerAttention(config, layer_id)
        self.intermediate = LongformerIntermediate(config)
        self.output = LongformerOutput(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, is_index_masked=None, is_index_global_attn=None, is_global_attn=None, output_attentions=False):
        if False:
            return 10
        self_attn_outputs = self.attention(hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn, output_attentions=output_attentions)
        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]
        layer_output = apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output)
        outputs = (layer_output,) + outputs
        return outputs

    def ff_chunk(self, attn_output):
        if False:
            for i in range(10):
                print('nop')
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output

class LongformerEncoder(nn.Module):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LongformerLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None, padding_len=0, output_attentions=False, output_hidden_states=False, return_dict=True):
        if False:
            for i in range(10):
                print('nop')
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_global_attentions = () if output_attentions and is_global_attn else None
        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layer), f'The head_mask should be specified for {len(self.layer)} layers, but it is for {head_mask.size()[0]}.'
        for (idx, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None, is_index_masked, is_index_global_attn, is_global_attn, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)
                if is_global_attn:
                    all_global_attentions = all_global_attentions + (layer_outputs[2].transpose(2, 3),)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states = hidden_states[:, :hidden_states.shape[1] - padding_len]
        if output_hidden_states:
            all_hidden_states = tuple([state[:, :state.shape[1] - padding_len] for state in all_hidden_states])
        if output_attentions:
            all_attentions = tuple([state[:, :, :state.shape[2] - padding_len, :] for state in all_attentions])
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions, all_global_attentions] if v is not None))
        return LongformerBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions, global_attentions=all_global_attentions)

class LongformerPooler(nn.Module):

    def __init__(self, config):
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class LongformerLMHead(nn.Module):
    """Longformer Head for masked language modeling."""

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        if False:
            while True:
                i = 10
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        if False:
            for i in range(10):
                print('nop')
        if self.decoder.bias.device.type == 'meta':
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias

class LongformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LongformerConfig
    base_model_prefix = 'longformer'
    supports_gradient_checkpointing = True
    _no_split_modules = ['LongformerSelfAttention']

    def _init_weights(self, module):
        if False:
            return 10
        'Initialize the weights'
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
LONGFORMER_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`LongformerConfig`]): Model configuration class with all the parameters of the\n            model. Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
LONGFORMER_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        global_attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):\n            Mask to decide the attention given on each token, local attention or global attention. Tokens with global\n            attention attends to all other tokens, and all other tokens attend to them. This is important for\n            task-specific finetuning because it makes the model more flexible at representing the task. For example,\n            for classification, the <s> token should be given global attention. For QA, all question tokens should also\n            have global attention. Please refer to the [Longformer paper](https://arxiv.org/abs/2004.05150) for more\n            details. Mask values selected in `[0, 1]`:\n\n            - 0 for local attention (a sliding window attention),\n            - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).\n\n        head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        decoder_head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):\n            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):\n            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,\n            config.max_position_embeddings - 1]`.\n\n            [What are position IDs?](../glossary#position-ids)\n        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare Longformer Model outputting raw hidden-states without any specific head on top.', LONGFORMER_START_DOCSTRING)
class LongformerModel(LongformerPreTrainedModel):
    """
    This class copied code from [`RobertaModel`] and overwrote standard self-attention with longformer self-attention
    to provide the ability to process long sequences following the self-attention approach described in [Longformer:
    the Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, and Arman Cohan.
    Longformer self-attention combines a local (sliding window) and global attention to extend to long documents
    without the O(n^2) increase in memory and compute.

    The self-attention module `LongformerSelfAttention` implemented here supports the combination of local and global
    attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and dilated
    attention are more relevant for autoregressive language modeling than finetuning on downstream tasks. Future
    release will add support for autoregressive attention, but the support for dilated attention requires a custom CUDA
    kernel to be memory and compute efficient.

    """

    def __init__(self, config, add_pooling_layer=True):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.config = config
        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, '`config.attention_window` has to be an even value'
            assert config.attention_window > 0, '`config.attention_window` has to be positive'
            config.attention_window = [config.attention_window] * config.num_hidden_layers
        else:
            assert len(config.attention_window) == config.num_hidden_layers, f'`len(config.attention_window)` should equal `config.num_hidden_layers`. Expected {config.num_hidden_layers}, given {len(config.attention_window)}'
        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)
        self.pooler = LongformerPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        if False:
            print('Hello World!')
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if False:
            print('Hello World!')
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        if False:
            return 10
        '\n        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base\n        class PreTrainedModel\n        '
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _pad_to_window_size(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, position_ids: torch.Tensor, inputs_embeds: torch.Tensor, pad_token_id: int):
        if False:
            while True:
                i = 10
        'A helper function to pad tokens and mask to work with implementation of Longformer self-attention.'
        attention_window = self.config.attention_window if isinstance(self.config.attention_window, int) else max(self.config.attention_window)
        assert attention_window % 2 == 0, f'`attention_window` should be an even value. Given {attention_window}'
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        (batch_size, seq_len) = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.info(f'Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of `config.attention_window`: {attention_window}')
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full((batch_size, padding_len), self.config.pad_token_id, dtype=torch.long)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)
            attention_mask = nn.functional.pad(attention_mask, (0, padding_len), value=0)
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)
        return (padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds)

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        if False:
            while True:
                i = 10
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            attention_mask = global_attention_mask + 1
        return attention_mask

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=LongformerBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, global_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LongformerBaseModelOutputWithPooling]:
        if False:
            print('Hello World!')
        '\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> import torch\n        >>> from transformers import LongformerModel, AutoTokenizer\n\n        >>> model = LongformerModel.from_pretrained("allenai/longformer-base-4096")\n        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")\n\n        >>> SAMPLE_TEXT = " ".join(["Hello world! "] * 1000)  # long input document\n        >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1\n\n        >>> attention_mask = torch.ones(\n        ...     input_ids.shape, dtype=torch.long, device=input_ids.device\n        ... )  # initialize to local attention\n        >>> global_attention_mask = torch.zeros(\n        ...     input_ids.shape, dtype=torch.long, device=input_ids.device\n        ... )  # initialize to global attention to be deactivated for all tokens\n        >>> global_attention_mask[\n        ...     :,\n        ...     [\n        ...         1,\n        ...         4,\n        ...         21,\n        ...     ],\n        ... ] = 1  # Set global attention to random tokens for the sake of this example\n        >>> # Usually, set global attention based on the task. For example,\n        >>> # classification: the <s> token\n        >>> # QA: question tokens\n        >>> # LM: potentially on the beginning of sentences and paragraphs\n        >>> outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)\n        >>> sequence_output = outputs.last_hidden_state\n        >>> pooled_output = outputs.pooler_output\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)
        (padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds) = self._pad_to_window_size(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, pad_token_id=self.config.pad_token_id)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[:, 0, 0, :]
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, padding_len=padding_len, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return LongformerBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, global_attentions=encoder_outputs.global_attentions)

@add_start_docstrings('Longformer Model with a `language modeling` head on top.', LONGFORMER_START_DOCSTRING)
class LongformerForMaskedLM(LongformerPreTrainedModel):
    _tied_weights_keys = ['lm_head.decoder']

    def __init__(self, config):
        if False:
            return 10
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        if False:
            i = 10
            return i + 15
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        if False:
            for i in range(10):
                print('nop')
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=LongformerMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, global_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LongformerMaskedLMOutput]:
        if False:
            while True:
                i = 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,\n            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the\n            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`\n        kwargs (`Dict[str, any]`, optional, defaults to *{}*):\n            Used to hide legacy arguments that have been deprecated.\n\n        Returns:\n\n        Mask filling example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, LongformerForMaskedLM\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")\n        >>> model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")\n        ```\n\n        Let\'s try a very long input.\n\n        ```python\n        >>> TXT = (\n        ...     "My friends are <mask> but they eat too many carbs."\n        ...     + " That\'s why I decide not to eat with them." * 300\n        ... )\n        >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]\n        >>> logits = model(input_ids).logits\n\n        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()\n        >>> probs = logits[0, masked_index].softmax(dim=0)\n        >>> values, predictions = probs.topk(5)\n\n        >>> tokenizer.decode(predictions).split()\n        [\'healthy\', \'skinny\', \'thin\', \'good\', \'vegetarian\']\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, head_mask=head_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return LongformerMaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions, global_attentions=outputs.global_attentions)

@add_start_docstrings('\n    Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', LONGFORMER_START_DOCSTRING)
class LongformerForSequenceClassification(LongformerPreTrainedModel):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = LongformerClassificationHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='jpwahle/longformer-base-plagiarism-detection', output_type=LongformerSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output="'ORIGINAL'", expected_loss=5.44)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, global_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LongformerSequenceClassifierOutput]:
        if False:
            i = 10
            return i + 15
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,\n            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If\n            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if global_attention_mask is None:
            logger.info('Initializing global attention on CLS token...')
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
        outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, head_mask=head_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return LongformerSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, global_attentions=outputs.global_attentions)

class LongformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output

@add_start_docstrings('\n    Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD /\n    TriviaQA (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', LONGFORMER_START_DOCSTRING)
class LongformerForQuestionAnswering(LongformerPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=LongformerQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, global_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, start_positions: Optional[torch.Tensor]=None, end_positions: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LongformerQuestionAnsweringModelOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the start of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for position (index) of the end of the labelled span for computing the token classification loss.\n            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence\n            are not taken into account for computing the loss.\n\n        Returns:\n\n        Examples:\n\n        ```python\n        >>> from transformers import AutoTokenizer, LongformerForQuestionAnswering\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")\n        >>> model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")\n\n        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"\n        >>> encoding = tokenizer(question, text, return_tensors="pt")\n        >>> input_ids = encoding["input_ids"]\n\n        >>> # default is local attention everywhere\n        >>> # the forward method will automatically set global attention on question tokens\n        >>> attention_mask = encoding["attention_mask"]\n\n        >>> outputs = model(input_ids, attention_mask=attention_mask)\n        >>> start_logits = outputs.start_logits\n        >>> end_logits = outputs.end_logits\n        >>> all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())\n\n        >>> answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]\n        >>> answer = tokenizer.decode(\n        ...     tokenizer.convert_tokens_to_ids(answer_tokens)\n        ... )  # remove space prepending space token\n        ```'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if global_attention_mask is None:
            if input_ids is None:
                logger.warning('It is not possible to automatically generate the `global_attention_mask` because input_ids is None. Please make sure that it is correctly set.')
            else:
                global_attention_mask = _compute_global_attention_mask(input_ids, self.config.sep_token_id)
        outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, head_mask=head_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        (start_logits, end_logits) = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return LongformerQuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, global_attentions=outputs.global_attentions)

@add_start_docstrings('\n    Longformer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.\n    for Named-Entity-Recognition (NER) tasks.\n    ', LONGFORMER_START_DOCSTRING)
class LongformerForTokenClassification(LongformerPreTrainedModel):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='brad1141/Longformer-finetuned-norm', output_type=LongformerTokenClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output="['Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence']", expected_loss=0.63)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, global_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LongformerTokenClassifierOutput]:
        if False:
            return 10
        '\n        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.\n        '
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, head_mask=head_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return LongformerTokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, global_attentions=outputs.global_attentions)

@add_start_docstrings('\n    Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and\n    a softmax) e.g. for RocStories/SWAG tasks.\n    ', LONGFORMER_START_DOCSTRING)
class LongformerForMultipleChoice(LongformerPreTrainedModel):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=LongformerMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, global_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LongformerMultipleChoiceModelOutput]:
        if False:
            for i in range(10):
                print('nop')
        '\n        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):\n            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,\n            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See\n            `input_ids` above)\n        '
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if global_attention_mask is None and input_ids is not None:
            logger.info('Initializing global attention on multiple choice...')
            global_attention_mask = torch.stack([_compute_global_attention_mask(input_ids[:, i], self.config.sep_token_id, before_sep_token=False) for i in range(num_choices)], dim=1)
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_global_attention_mask = global_attention_mask.view(-1, global_attention_mask.size(-1)) if global_attention_mask is not None else None
        flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None
        outputs = self.longformer(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask, global_attention_mask=flat_global_attention_mask, head_mask=head_mask, inputs_embeds=flat_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(reshaped_logits.device)
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return LongformerMultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, global_attentions=outputs.global_attentions)
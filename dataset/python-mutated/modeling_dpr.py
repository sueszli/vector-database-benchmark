""" PyTorch DPR model for Open Domain Question Answering."""
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..bert.modeling_bert import BertModel
from .configuration_dpr import DPRConfig
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'DPRConfig'
_CHECKPOINT_FOR_DOC = 'facebook/dpr-ctx_encoder-single-nq-base'
DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/dpr-ctx_encoder-single-nq-base', 'facebook/dpr-ctx_encoder-multiset-base']
DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/dpr-question_encoder-single-nq-base', 'facebook/dpr-question_encoder-multiset-base']
DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = ['facebook/dpr-reader-single-nq-base', 'facebook/dpr-reader-multiset-base']

@dataclass
class DPRContextEncoderOutput(ModelOutput):
    """
    Class for outputs of [`DPRQuestionEncoder`].

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`):
            The DPR encoder outputs the *pooler_output* that corresponds to the context representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed contexts for nearest neighbors queries with questions embeddings.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class DPRQuestionEncoderOutput(ModelOutput):
    """
    Class for outputs of [`DPRQuestionEncoder`].

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`):
            The DPR encoder outputs the *pooler_output* that corresponds to the question representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed questions for nearest neighbors queries with context embeddings.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class DPRReaderOutput(ModelOutput):
    """
    Class for outputs of [`DPRQuestionEncoder`].

    Args:
        start_logits (`torch.FloatTensor` of shape `(n_passages, sequence_length)`):
            Logits of the start index of the span for each passage.
        end_logits (`torch.FloatTensor` of shape `(n_passages, sequence_length)`):
            Logits of the end index of the span for each passage.
        relevance_logits (`torch.FloatTensor` of shape `(n_passages, )`):
            Outputs of the QA classifier of the DPRReader that corresponds to the scores of each passage to answer the
            question, compared to all the other passages.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    start_logits: torch.FloatTensor
    end_logits: torch.FloatTensor = None
    relevance_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class DPRPreTrainedModel(PreTrainedModel):

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

class DPREncoder(DPRPreTrainedModel):
    base_model_prefix = 'bert_model'

    def __init__(self, config: DPRConfig):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.bert_model = BertModel(config, add_pooling_layer=False)
        if self.bert_model.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = nn.Linear(self.bert_model.config.hidden_size, config.projection_dim)
        self.post_init()

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor]=None, token_type_ids: Optional[Tensor]=None, inputs_embeds: Optional[Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=False) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:
        if False:
            print('Hello World!')
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)
        if not return_dict:
            return (sequence_output, pooled_output) + outputs[2:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    @property
    def embeddings_size(self) -> int:
        if False:
            return 10
        if self.projection_dim > 0:
            return self.encode_proj.out_features
        return self.bert_model.config.hidden_size

class DPRSpanPredictor(DPRPreTrainedModel):
    base_model_prefix = 'encoder'

    def __init__(self, config: DPRConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.encoder = DPREncoder(config)
        self.qa_outputs = nn.Linear(self.encoder.embeddings_size, 2)
        self.qa_classifier = nn.Linear(self.encoder.embeddings_size, 1)
        self.post_init()

    def forward(self, input_ids: Tensor, attention_mask: Tensor, inputs_embeds: Optional[Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=False) -> Union[DPRReaderOutput, Tuple[Tensor, ...]]:
        if False:
            return 10
        (n_passages, sequence_length) = input_ids.size() if input_ids is not None else inputs_embeds.size()[:2]
        outputs = self.encoder(input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        (start_logits, end_logits) = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])
        start_logits = start_logits.view(n_passages, sequence_length)
        end_logits = end_logits.view(n_passages, sequence_length)
        relevance_logits = relevance_logits.view(n_passages)
        if not return_dict:
            return (start_logits, end_logits, relevance_logits) + outputs[2:]
        return DPRReaderOutput(start_logits=start_logits, end_logits=end_logits, relevance_logits=relevance_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class DPRPretrainedContextEncoder(DPRPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = 'ctx_encoder'

class DPRPretrainedQuestionEncoder(DPRPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = 'question_encoder'

class DPRPretrainedReader(DPRPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = 'span_predictor'
DPR_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`DPRConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'
DPR_ENCODERS_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. To match pretraining, DPR input sequence should be\n            formatted with [CLS] and [SEP] tokens as follows:\n\n            (a) For sequence pairs (for a pair title+text for example):\n\n            ```\n            tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]\n            token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1\n            ```\n\n            (b) For single sequences (for a question for example):\n\n            ```\n            tokens:         [CLS] the dog is hairy . [SEP]\n            token_type_ids:   0   0   0   0  0     0   0\n            ```\n\n            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right\n            rather than the left.\n\n            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n            [`PreTrainedTokenizer.__call__`] for details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,\n            1]`:\n\n            - 0 corresponds to a *sentence A* token,\n            - 1 corresponds to a *sentence B* token.\n\n            [What are token type IDs?](../glossary#token-type-ids)\n        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"
DPR_READER_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`Tuple[torch.LongTensor]` of shapes `(n_passages, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. It has to be a sequence triplet with 1) the question\n            and 2) the passages titles and 3) the passages texts To match pretraining, DPR `input_ids` sequence should\n            be formatted with [CLS] and [SEP] with the format:\n\n                `[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>`\n\n            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right\n            rather than the left.\n\n            Indices can be obtained using [`DPRReaderTokenizer`]. See this class documentation for more details.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.FloatTensor` of shape `(n_passages, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        inputs_embeds (`torch.FloatTensor` of shape `(n_passages, sequence_length, hidden_size)`, *optional*):\n            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This\n            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the\n            model's internal embedding lookup matrix.\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        return_dict (`bool`, *optional*):\n            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.\n"

@add_start_docstrings('The bare DPRContextEncoder transformer outputting pooler outputs as context representations.', DPR_START_DOCSTRING)
class DPRContextEncoder(DPRPretrainedContextEncoder):

    def __init__(self, config: DPRConfig):
        if False:
            print('Hello World!')
        super().__init__(config)
        self.config = config
        self.ctx_encoder = DPREncoder(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRContextEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[Tensor]=None, attention_mask: Optional[Tensor]=None, token_type_ids: Optional[Tensor]=None, inputs_embeds: Optional[Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[DPRContextEncoderOutput, Tuple[Tensor, ...]]:
        if False:
            i = 10
            return i + 15
        '\n        Return:\n\n        Examples:\n\n        ```python\n        >>> from transformers import DPRContextEncoder, DPRContextEncoderTokenizer\n\n        >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")\n        >>> model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")\n        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]\n        >>> embeddings = model(input_ids).pooler_output\n        ```'
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device) if input_ids is None else input_ids != self.config.pad_token_id
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        outputs = self.ctx_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return outputs[1:]
        return DPRContextEncoderOutput(pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.', DPR_START_DOCSTRING)
class DPRQuestionEncoder(DPRPretrainedQuestionEncoder):

    def __init__(self, config: DPRConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.config = config
        self.question_encoder = DPREncoder(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRQuestionEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[Tensor]=None, attention_mask: Optional[Tensor]=None, token_type_ids: Optional[Tensor]=None, inputs_embeds: Optional[Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[DPRQuestionEncoderOutput, Tuple[Tensor, ...]]:
        if False:
            print('Hello World!')
        '\n        Return:\n\n        Examples:\n\n        ```python\n        >>> from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer\n\n        >>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")\n        >>> model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")\n        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]\n        >>> embeddings = model(input_ids).pooler_output\n        ```\n        '
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
            attention_mask = torch.ones(input_shape, device=device) if input_ids is None else input_ids != self.config.pad_token_id
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        outputs = self.question_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return outputs[1:]
        return DPRQuestionEncoderOutput(pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings('The bare DPRReader transformer outputting span predictions.', DPR_START_DOCSTRING)
class DPRReader(DPRPretrainedReader):

    def __init__(self, config: DPRConfig):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self.config = config
        self.span_predictor = DPRSpanPredictor(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(DPR_READER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DPRReaderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[Tensor]=None, attention_mask: Optional[Tensor]=None, inputs_embeds: Optional[Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[DPRReaderOutput, Tuple[Tensor, ...]]:
        if False:
            while True:
                i = 10
        '\n        Return:\n\n        Examples:\n\n        ```python\n        >>> from transformers import DPRReader, DPRReaderTokenizer\n\n        >>> tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")\n        >>> model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")\n        >>> encoded_inputs = tokenizer(\n        ...     questions=["What is love ?"],\n        ...     titles=["Haddaway"],\n        ...     texts=["\'What Is Love\' is a song recorded by the artist Haddaway"],\n        ...     return_tensors="pt",\n        ... )\n        >>> outputs = model(**encoded_inputs)\n        >>> start_logits = outputs.start_logits\n        >>> end_logits = outputs.end_logits\n        >>> relevance_logits = outputs.relevance_logits\n        ```\n        '
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
        return self.span_predictor(input_ids, attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
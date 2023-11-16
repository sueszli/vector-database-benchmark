"""RAG model implementation."""
import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'RagConfig'

@dataclass
class RetrievAugLMMarginOutput(ModelOutput):
    """
    Base class for retriever augmented marginalized models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`.
        past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        retrieved_doc_embeds (`torch.FloatTensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output_retrieved=True*):
            Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
            the `doc_scores`.
        retrieved_doc_ids (`torch.LongTensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output_retrieved=True*):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.
        question_encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    doc_scores: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None
    retrieved_doc_ids: Optional[torch.LongTensor] = None
    context_input_ids: Optional[torch.LongTensor] = None
    context_attention_mask: Optional[torch.LongTensor] = None
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class RetrievAugLMOutput(ModelOutput):
    """
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`.
        past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        retrieved_doc_embeds (`torch.FloatTensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output_retrieved=True*):
            Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
            the `doc_scores`.
        retrieved_doc_ids (`torch.LongTensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output_retrieved=True*):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.
        question_encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """
    logits: torch.FloatTensor = None
    doc_scores: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None
    retrieved_doc_ids: Optional[torch.LongTensor] = None
    context_input_ids: Optional[torch.LongTensor] = None
    context_attention_mask: Optional[torch.LongTensor] = None
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

class RagPreTrainedModel(PreTrainedModel):
    """
    RAG models were released with the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP
    Tasks](https://arxiv.org/abs/2005.11401) by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    RAG is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.

    """
    config_class = RagConfig
    base_model_prefix = 'rag'

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs['_fast_init'] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained_question_encoder_generator(cls, question_encoder_pretrained_model_name_or_path: str=None, generator_pretrained_model_name_or_path: str=None, retriever: RagRetriever=None, **kwargs) -> PreTrainedModel:
        if False:
            while True:
                i = 10
        '\n        Instantiates an question encoder and a generator from one or two base classes of the library from pretrained\n        model checkpoints.\n\n        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train\n        the model, you need to first set it back in training mode with `model.train()`.\n\n        Params:\n            question_encoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):\n                Information necessary to initiate the question encoder. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In\n                      this case, `from_tf` should be set to `True` and a configuration object should be provided as\n                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a\n                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.\n\n            generator_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):\n                Information necessary to initiate the generator. Can be either:\n\n                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.\n                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a\n                      user or organization name, like `dbmdz/bert-base-german-cased`.\n                    - A path to a *directory* containing model weights saved using\n                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.\n                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In\n                      this case, `from_tf` should be set to `True` and a configuration object should be provided as\n                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a\n                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.\n\n            model_args (remaining positional arguments, *optional*):\n                All remaining positional arguments will be passed to the underlying model\'s `__init__` method.\n            retriever ([`RagRetriever`], *optional*):\n                The retriever to use.\n            kwwargs (remaining dictionary of keyword arguments, *optional*):\n                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,\n                `output_attentions=True`).\n\n                - To update the question_encoder configuration, use the prefix *question_encoder_* for each\n                  configuration parameter.\n                - To update the generator configuration, use the prefix *generator_* for each configuration parameter.\n                - To update the parent model configuration, do not use a prefix for each configuration parameter.\n\n                Behaves differently depending on whether a `config` is provided or automatically loaded.\n\n        Example:\n\n        ```python\n        >>> from transformers import RagModel\n\n        >>> # initialize a RAG from two pretrained models.\n        >>> model = RagModel.from_pretrained_question_encoder_generator(\n        ...     "facebook/dpr-question_encoder-single-nq-base", "t5-small"\n        ... )\n        >>> # saving model after fine-tuning\n        >>> model.save_pretrained("./rag")\n        >>> # load fine-tuned model\n        >>> model = RagModel.from_pretrained("./rag")\n        ```'
        kwargs_question_encoder = {argument[len('question_encoder_'):]: value for (argument, value) in kwargs.items() if argument.startswith('question_encoder_')}
        kwargs_generator = {argument[len('generator_'):]: value for (argument, value) in kwargs.items() if argument.startswith('generator_')}
        for key in kwargs_question_encoder.keys():
            del kwargs['question_encoder_' + key]
        for key in kwargs_generator.keys():
            del kwargs['generator_' + key]
        question_encoder = kwargs_question_encoder.pop('model', None)
        if question_encoder is None:
            assert question_encoder_pretrained_model_name_or_path is not None, 'If `model` is not defined as an argument, a `question_encoder_pretrained_model_name_or_path` has to be defined'
            from ..auto.modeling_auto import AutoModel
            if 'config' not in kwargs_question_encoder:
                from ..auto.configuration_auto import AutoConfig
                (question_encoder_config, kwargs_question_encoder) = AutoConfig.from_pretrained(question_encoder_pretrained_model_name_or_path, **kwargs_question_encoder, return_unused_kwargs=True)
                kwargs_question_encoder['config'] = question_encoder_config
            question_encoder = AutoModel.from_pretrained(question_encoder_pretrained_model_name_or_path, **kwargs_question_encoder)
        generator = kwargs_generator.pop('model', None)
        if generator is None:
            assert generator_pretrained_model_name_or_path is not None, 'If `generator_model` is not defined as an argument, a `generator_pretrained_model_name_or_path` has to be defined'
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM
            if 'config' not in kwargs_generator:
                from ..auto.configuration_auto import AutoConfig
                (generator_config, kwargs_generator) = AutoConfig.from_pretrained(generator_pretrained_model_name_or_path, **kwargs_generator, return_unused_kwargs=True)
                kwargs_generator['config'] = generator_config
            generator = AutoModelForSeq2SeqLM.from_pretrained(generator_pretrained_model_name_or_path, **kwargs_generator)
        config = kwargs.get('config', None)
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(question_encoder.config, generator.config, **kwargs)
        return cls(question_encoder=question_encoder, generator=generator, config=config, retriever=retriever)
RAG_START_DOCSTRING = '\n\n    RAG is a seq2seq model which encapsulates two core components: a question encoder and a generator. During a forward\n    pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context\n    documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.\n\n    The question encoder can be any *autoencoding* model, preferably [`DPRQuestionEncoder`], and the generator can be\n    any *seq2seq* model, preferably [`BartForConditionalGeneration`].\n\n    The model can be initialized with a [`RagRetriever`] for end-to-end generation or used in combination with the\n    outputs of a retriever in multiple steps---see examples for more details. The model is compatible any\n    *autoencoding* model as the `question_encoder` and any *seq2seq* model with language model head as the `generator`.\n    It has been tested with [`DPRQuestionEncoder`] as the `question_encoder` and [`BartForConditionalGeneration`] or\n    [`T5ForConditionalGeneration`] as the `generator`.\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n\n    Args:\n        config ([`RagConfig`]):\n            Model configuration class with all the parameters of the model. Initializing with a config file does not\n            load the weights associated with the model, only the configuration. Check out the\n            [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n        question_encoder ([`PreTrainedModel`]):\n            An encoder model compatible with the faiss index encapsulated by the `retriever`.\n        generator ([`PreTrainedModel`]):\n            A seq2seq model used as the generator in the RAG architecture.\n        retriever ([`RagRetriever`]):\n            A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.\n'
RAG_FORWARD_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary. [`RagConfig`], used to initialize the model, specifies\n            which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to\n            obtain the indices.\n\n            [What are input IDs?](../glossary#input-ids)\n        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            [What are attention masks?](../glossary#attention-mask)\n        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*)\n            Tuple consists of (`generator_enc_last_hidden_state`, *optional*: `generator_enc_hidden_states`,\n            *optional*: `generator_enc_attentions`). `generator_enc_last_hidden_state` of shape `(batch_size, n_docs *\n            sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the\n            generator's encoder.\n\n            Used by the ([`RagModel`]) model during decoding.\n        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):\n            Provide for generation tasks. `None` by default, construct as per instructions for the generator model\n            you're using with your RAG instance.\n        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size,  target_sequence_length)`, *optional*):\n            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also\n            be used by default.\n        past_key_values (`tuple(tuple(torch.FloatTensor))`):\n            Tuple consists of two elements: `encoder_outputs` of the RAG model (see `encoder_outputs`) and\n            `past_key_values` of the underlying generator. Can be used to speed up decoding. `past_key_values` are used\n            in the ([`RagTokenForGeneration`]) model during decoding.\n        doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):\n            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and\n            `question_encoder_last_hidden_state`. If the model has is not initialized with a `retriever` `doc_scores`\n            has to be provided to the forward pass. `doc_scores` can be computed via\n            `question_encoder_last_hidden_state` and `retrieved_doc_embeds`, see examples for more information.\n        context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):\n            Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the\n            retriever. If the model was not initialized with a `retriever` ``context_input_ids` has to be provided to\n            the forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].\n        context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`,*optional*, returned when *output_retrieved=True*):\n            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the\n            retriever. If the model has is not initialized with a `retriever` `context_attention_mask` has to be\n            provided to the forward pass. `context_attention_mask` are returned by [`~RagRetriever.__call__`].\n        use_cache (`bool`, *optional*, defaults to `True`):\n            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see\n            `past_key_values`).\n        output_attentions (`bool`, *optional*):\n            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned\n            tensors for more detail.\n        output_hidden_states (`bool`, *optional*):\n            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for\n            more detail.\n        output_retrieved(`bool`, *optional*):\n            Whether or not to return the `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and\n            `context_attention_mask`. See returned tensors for more detail.\n        n_docs (`int`, *optional*, defaults to `config.n_docs``)\n            Number of documents to retrieve and/or number of documents for which to generate an answer.\n"

@add_start_docstrings_to_model_forward(RAG_START_DOCSTRING)
class RagModel(RagPreTrainedModel):

    def __init__(self, config: Optional[PretrainedConfig]=None, question_encoder: Optional[PreTrainedModel]=None, generator: Optional[PreTrainedModel]=None, retriever: Optional[RagRetriever]=None, **kwargs):
        if False:
            return 10
        assert config is not None or (question_encoder is not None and generator is not None), 'Either a configuration or an question_encoder and a generator has to be provided.'
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(question_encoder.config, generator.config, **kwargs)
        else:
            assert isinstance(config, self.config_class), f'config: {config} has to be of type {self.config_class}'
        super().__init__(config)
        if question_encoder is None:
            from ..auto.modeling_auto import AutoModel
            question_encoder = AutoModel.from_config(config.question_encoder)
        if generator is None:
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM
            generator = AutoModelForSeq2SeqLM.from_config(config.generator)
        self.retriever = retriever
        if self.retriever is not None:
            assert isinstance(retriever, RagRetriever), f'`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`'
            self.retriever = retriever
        self.question_encoder = question_encoder
        self.generator = generator
        self.ctx_encoder = None
        self.context_encoder_training = False

    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, doc_scores: Optional[torch.FloatTensor]=None, context_input_ids: Optional[torch.LongTensor]=None, context_attention_mask: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_retrieved: Optional[bool]=None, n_docs: Optional[int]=None) -> Union[Tuple[torch.Tensor], RetrievAugLMOutput]:
        if False:
            print('Hello World!')
        '\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, RagRetriever, RagModel\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")\n        >>> retriever = RagRetriever.from_pretrained(\n        ...     "facebook/rag-token-base", index_name="exact", use_dummy_dataset=True\n        ... )\n        >>> # initialize with RagRetriever to do everything in one forward call\n        >>> model = RagModel.from_pretrained("facebook/rag-token-base", retriever=retriever)\n\n        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")\n        >>> outputs = model(input_ids=inputs["input_ids"])\n        ```'
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_retrieved = output_retrieved if output_retrieved is not None else self.config.output_retrieved
        has_to_retrieve = self.retriever is not None and (context_input_ids is None or context_attention_mask is None or doc_scores is None) and (encoder_outputs is None)
        if encoder_outputs is None:
            if has_to_retrieve:
                question_enc_outputs = self.question_encoder(input_ids, attention_mask=attention_mask, return_dict=True)
                question_encoder_last_hidden_state = question_enc_outputs[0]
                retriever_outputs = self.retriever(input_ids, question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(), prefix=self.generator.config.prefix, n_docs=n_docs, return_tensors='pt')
                if self.context_encoder_training:
                    (context_input_ids, context_attention_mask, retrieved_doc_embeds, retrived_doc_input_ids, retrived_doc_attention_mask, retrieved_doc_ids) = (retriever_outputs['context_input_ids'], retriever_outputs['context_attention_mask'], retriever_outputs['retrieved_doc_embeds'], retriever_outputs['tokenized_doc_ids'], retriever_outputs['tokenized_doc_attention_mask'], retriever_outputs['doc_ids'])
                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)
                    retrived_doc_input_ids = retrived_doc_input_ids.to(input_ids)
                    retrived_doc_attention_mask = retrived_doc_attention_mask.to(input_ids)
                    retrieved_doc_embeds = self.ctx_encoder(retrived_doc_input_ids, attention_mask=retrived_doc_attention_mask, return_dict=True).pooler_output
                    retrieved_doc_embeds = retrieved_doc_embeds.view(-1, n_docs, question_encoder_last_hidden_state.shape[1])
                    doc_scores = torch.bmm(question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
                else:
                    (context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids) = (retriever_outputs['context_input_ids'], retriever_outputs['context_attention_mask'], retriever_outputs['retrieved_doc_embeds'], retriever_outputs['doc_ids'])
                    retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)
                    doc_scores = torch.bmm(question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
            else:
                assert context_input_ids is not None, 'Make sure that `context_input_ids` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function.'
                assert context_attention_mask is not None, 'Make sure that `context_attention_mask` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function.'
                assert doc_scores is not None, 'Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function.'
        assert doc_scores is not None, 'Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function.'
        assert doc_scores.shape[1] % n_docs == 0, f' The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}.'
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)
        gen_outputs = self.generator(input_ids=context_input_ids, attention_mask=context_attention_mask, encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, return_dict=True)
        if not has_to_retrieve:
            question_encoder_last_hidden_state = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions
        if not has_to_retrieve or not output_retrieved:
            context_input_ids = (None,)
            context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        return RetrievAugLMOutput(logits=gen_outputs.logits, doc_scores=doc_scores, past_key_values=gen_outputs.past_key_values, context_input_ids=context_input_ids, context_attention_mask=context_attention_mask, retrieved_doc_embeds=retrieved_doc_embeds, retrieved_doc_ids=retrieved_doc_ids, question_encoder_last_hidden_state=question_encoder_last_hidden_state, question_enc_hidden_states=question_enc_hidden_states, question_enc_attentions=question_enc_attentions, generator_enc_last_hidden_state=gen_outputs.encoder_last_hidden_state, generator_enc_hidden_states=gen_outputs.encoder_hidden_states, generator_enc_attentions=gen_outputs.encoder_attentions, generator_dec_hidden_states=gen_outputs.decoder_hidden_states, generator_dec_attentions=gen_outputs.decoder_attentions, generator_cross_attentions=gen_outputs.cross_attentions)

@add_start_docstrings_to_model_forward('\n    A RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.\n    ', RAG_START_DOCSTRING)
class RagSequenceForGeneration(RagPreTrainedModel):

    def __init__(self, config: Optional[PretrainedConfig]=None, question_encoder: Optional[PreTrainedModel]=None, generator: Optional[PreTrainedModel]=None, retriever: Optional[RagRetriever]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert config is not None or (question_encoder is not None and generator is not None), 'Either a configuration or an encoder and a generator has to be provided.'
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(question_encoder.config, generator.config, **kwargs)
        super().__init__(config)
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    def set_retriever(self, retriever: RagRetriever):
        if False:
            while True:
                i = 10
        self.rag.retriever = retriever

    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        if False:
            i = 10
            return i + 15
        self.rag.context_encoder_training = True
        self.rag.ctx_encoder = ctx_encoder

    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, context_input_ids: Optional[torch.LongTensor]=None, context_attention_mask: Optional[torch.LongTensor]=None, doc_scores: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_retrieved: Optional[bool]=None, exclude_bos_score: Optional[bool]=None, reduce_loss: Optional[bool]=None, labels: Optional[torch.LongTensor]=None, n_docs: Optional[int]=None, **kwargs) -> RetrievAugLMMarginOutput:
        if False:
            i = 10
            return i + 15
        '\n        exclude_bos_score (`bool`, *optional*):\n            Only relevant if `labels` is passed. If `True`, the score of the BOS token is disregarded when computing\n            the loss.\n        reduce_loss (`bool`, *optional*):\n            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `torch.Tensor.sum`\n            operation.\n        kwargs (`Dict[str, any]`, optional, defaults to *{}*):\n             Legacy dictionary, which is required so that model can use *generate()* function.\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")\n        >>> retriever = RagRetriever.from_pretrained(\n        ...     "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True\n        ... )\n        >>> # initialize with RagRetriever to do everything in one forward call\n        >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)\n\n        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")\n        >>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")\n        >>> input_ids = inputs["input_ids"]\n        >>> labels = targets["input_ids"]\n        >>> outputs = model(input_ids=input_ids, labels=labels)\n\n        >>> # or use retriever separately\n        >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)\n        >>> # 1. Encode\n        >>> question_hidden_states = model.question_encoder(input_ids)[0]\n        >>> # 2. Retrieve\n        >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")\n        >>> doc_scores = torch.bmm(\n        ...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)\n        ... ).squeeze(1)\n        >>> # 3. Forward to generator\n        >>> outputs = model(\n        ...     context_input_ids=docs_dict["context_input_ids"],\n        ...     context_attention_mask=docs_dict["context_attention_mask"],\n        ...     doc_scores=doc_scores,\n        ...     decoder_input_ids=labels,\n        ... )\n        ```'
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        exclude_bos_score = exclude_bos_score if exclude_bos_score is not None else self.config.exclude_bos_score
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False
        outputs = self.rag(input_ids=input_ids, attention_mask=attention_mask, encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, context_input_ids=context_input_ids, context_attention_mask=context_attention_mask, doc_scores=doc_scores, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, output_retrieved=output_retrieved, n_docs=n_docs)
        loss = None
        if labels is not None:
            loss = self.get_nll(outputs.logits, outputs.doc_scores, decoder_input_ids, reduce_loss=reduce_loss, epsilon=self.config.label_smoothing, exclude_bos_score=exclude_bos_score, n_docs=n_docs)
        return RetrievAugLMMarginOutput(loss=loss, logits=outputs.logits, doc_scores=outputs.doc_scores, past_key_values=outputs.past_key_values, context_input_ids=outputs.context_input_ids, context_attention_mask=outputs.context_attention_mask, retrieved_doc_embeds=outputs.retrieved_doc_embeds, retrieved_doc_ids=outputs.retrieved_doc_ids, question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state, question_enc_hidden_states=outputs.question_enc_hidden_states, question_enc_attentions=outputs.question_enc_attentions, generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state, generator_enc_hidden_states=outputs.generator_enc_hidden_states, generator_enc_attentions=outputs.generator_enc_attentions, generator_dec_hidden_states=outputs.generator_dec_hidden_states, generator_dec_attentions=outputs.generator_dec_attentions, generator_cross_attentions=outputs.generator_cross_attentions)

    @property
    def retriever(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rag.retriever

    @property
    def generator(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rag.generator

    @property
    def question_encoder(self):
        if False:
            i = 10
            return i + 15
        return self.rag.question_encoder

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, context_input_ids: Optional[torch.LongTensor]=None, context_attention_mask: Optional[torch.LongTensor]=None, doc_scores: Optional[torch.FloatTensor]=None, do_deduplication: Optional[bool]=None, num_return_sequences: Optional[int]=None, num_beams: Optional[int]=None, n_docs: Optional[int]=None, **model_kwargs) -> torch.LongTensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Implements RAG sequence "thorough" decoding. Read the [`~generation.GenerationMixin.generate`]` documentation\n        for more information on how to set other generate input parameters.\n\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                The sequence used as a prompt for the generation. If `input_ids` is not passed, then\n                `context_input_ids` has to be provided.\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):\n                Input IDs post-processed from the retrieved documents and the question encoder input_ids by the\n                retriever.\n            context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):\n                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the\n                retriever.\n\n                If the model is not initialized with a `retriever` or `input_ids` is not given, `context_input_ids` and\n                `context_attention_mask` have to be provided to the forward pass. They are returned by\n                [`~RagRetriever.__call__`].\n            doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):\n                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and\n                `question_encoder_last_hidden_state`.\n\n                If the model is not initialized with a `retriever` or `input_ids` is not given, `doc_scores` has to be\n                provided to the forward pass. `doc_scores` are returned by [`~RagRetriever.__call__`].\n            do_deduplication (`bool`, *optional*):\n                Whether or not to deduplicate the generations from different context documents for a given input. Has\n                to be set to `False` if used while training with distributed backend.\n            num_return_sequences(`int`, *optional*, defaults to 1):\n                The number of independently computed returned sequences for each element in the batch. Note that this\n                is not the value we pass to the `generator`\'s `[`~generation.GenerationMixin.generate`]` function,\n                where we set `num_return_sequences` to `num_beams`.\n            num_beams (`int`, *optional*, defaults to 1):\n                Number of beams for beam search. 1 means no beam search.\n            n_docs (`int`, *optional*, defaults to `config.n_docs`)\n                Number of documents to retrieve and/or number of documents for which to generate an answer.\n            kwargs (`Dict[str, Any]`, *optional*):\n                Additional kwargs will be passed to [`~generation.GenerationMixin.generate`].\n\n        Return:\n            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated\n            sequences. The second dimension (sequence length) is either equal to `max_length` or shorter if all batches\n            finished early due to the `eos_token_id`.\n        '
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_deduplication = do_deduplication if do_deduplication is not None else self.config.do_deduplication
        num_doc_return_sequences = num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        assert input_ids is not None or context_input_ids is not None, ' At least one of input_ids or context_input_ids must be given'
        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            context_input_ids = self.retriever(input_ids, question_hidden_states.cpu().detach().to(torch.float32).numpy(), prefix=self.generator.config.prefix, n_docs=n_docs, return_tensors='pt')['context_input_ids']
            context_input_ids = context_input_ids.to(input_ids)
        hypos = []
        model_kwargs['num_beams'] = num_beams
        model_kwargs['num_return_sequences'] = num_beams
        model_kwargs['attention_mask'] = None
        batch_size = input_ids.shape[0] if input_ids is not None else context_input_ids.shape[0] // n_docs
        for index in range(batch_size):
            generator_input_ids = context_input_ids[index * n_docs:(index + 1) * n_docs]
            output_sequences = self.generator.generate(generator_input_ids, **model_kwargs)
            if do_deduplication:
                output_sequences = torch.stack(list({str(k.tolist()): k for k in output_sequences}.values()))
            num_candidates = output_sequences.shape[0]
            if input_ids is not None:
                new_input_ids = input_ids[index:index + 1].repeat(num_candidates, 1)
                outputs = self(new_input_ids, labels=output_sequences, exclude_bos_score=True)
            else:
                assert context_attention_mask is not None, 'Make sure that `context_attention_mask` are passed, if no `input_ids` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function.'
                assert doc_scores is not None, 'Make sure that `doc_scores` are passed, if no `input_ids` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function.'
                individual_input_ids = generator_input_ids.repeat(num_candidates, 1)
                individual_attention_mask = context_attention_mask[index * n_docs:(index + 1) * n_docs]
                individual_attention_mask = individual_attention_mask.repeat(num_candidates, 1)
                individual_doc_scores = doc_scores[index:index + 1, :]
                individual_doc_scores = individual_doc_scores.repeat(num_candidates, 1)
                outputs = self(context_input_ids=individual_input_ids, context_attention_mask=individual_attention_mask, doc_scores=individual_doc_scores, labels=output_sequences, exclude_bos_score=True)
            top_cand_inds = (-outputs['loss']).topk(num_doc_return_sequences)[1]
            hypos.append(output_sequences[top_cand_inds])
        return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id)

    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None):
        if False:
            for i in range(10):
                print('nop')
        target = torch.cat([target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1)
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            if False:
                i = 10
                return i + 15
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return (ll.squeeze(-1), smooth_obj.squeeze(-1))
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1))
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()
        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)
        (ll, smooth_obj) = _mask_pads(ll, smooth_obj)
        ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
        smooth_obj = smooth_obj.sum(2)
        ll = ll.logsumexp(1)
        smooth_obj = smooth_obj.logsumexp(1)
        nll_loss = -ll
        smooth_loss = -smooth_obj
        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        if False:
            print('Hello World!')
        output = tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])).fill_(pad_token_id)
        ind = 0
        for t in tensors:
            output[ind:ind + t.shape[0], :t.shape[1]] = t
            ind += t.shape[0]
        return output

@add_start_docstrings_to_model_forward('\n    A RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.\n    ', RAG_START_DOCSTRING)
class RagTokenForGeneration(RagPreTrainedModel):

    def __init__(self, config: Optional[PretrainedConfig]=None, question_encoder: Optional[PreTrainedModel]=None, generator: Optional[PreTrainedModel]=None, retriever: Optional[RagRetriever]=None, **kwargs):
        if False:
            return 10
        assert config is not None or (question_encoder is not None and generator is not None), 'Either a configuration or an encoder and a generator has to be provided.'
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(question_encoder.config, generator.config, **kwargs)
        super().__init__(config)
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    def set_retriever(self, retriever: RagRetriever):
        if False:
            return 10
        self.rag.retriever = retriever

    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        if False:
            print('Hello World!')
        self.rag.context_encoder_training = True
        self.rag.ctx_encoder = ctx_encoder

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, doc_scores=None, n_docs=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'doc_scores': doc_scores, 'context_attention_mask': attention_mask, 'decoder_input_ids': decoder_input_ids, 'past_key_values': past_key_values, 'use_cache': use_cache, 'do_marginalize': True, 'n_docs': n_docs}

    @property
    def retriever(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rag.retriever

    @property
    def generator(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rag.generator

    @property
    def question_encoder(self):
        if False:
            print('Hello World!')
        return self.rag.question_encoder

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if False:
            print('Hello World!')
        'Reorders cache for generation. BART-inspired but we need to take care of the extra dimension for docs'

        def _reorder_stacked(hidden_states, new_order):
            if False:
                print('Hello World!')
            n_docs = hidden_states.shape[0] // new_order.shape[0]
            hidden_states = hidden_states.view(-1, n_docs, *hidden_states.shape[1:])
            hidden_states = hidden_states.index_select(0, new_order)
            result = hidden_states.view(-1, *hidden_states.shape[2:])
            return result
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((_reorder_stacked(past_state, beam_idx.to(past_state.device)) for past_state in layer_past)),)
        return reordered_past

    def marginalize(self, seq_logits, doc_scores, n_docs=None):
        if False:
            return 10
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1))
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, context_input_ids: Optional[torch.LongTensor]=None, context_attention_mask: Optional[torch.LongTensor]=None, doc_scores: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_retrieved: Optional[bool]=None, do_marginalize: Optional[bool]=None, reduce_loss: Optional[bool]=None, labels: Optional[torch.LongTensor]=None, n_docs: Optional[int]=None, **kwargs) -> RetrievAugLMMarginOutput:
        if False:
            print('Hello World!')
        '\n        do_marginalize (`bool`, *optional*):\n            If `True`, the logits are marginalized over all documents by making use of\n            `torch.nn.functional.log_softmax`.\n        reduce_loss (`bool`, *optional*):\n            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `torch.Tensor.sum`\n            operation.\n        kwargs (`Dict[str, any]`, optional, defaults to *{}*):\n            Legacy dictionary, which is required so that model can use *generate()* function.\n\n        Returns:\n\n        Example:\n\n        ```python\n        >>> from transformers import AutoTokenizer, RagRetriever, RagTokenForGeneration\n        >>> import torch\n\n        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")\n        >>> retriever = RagRetriever.from_pretrained(\n        ...     "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True\n        ... )\n        >>> # initialize with RagRetriever to do everything in one forward call\n        >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)\n\n        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")\n        >>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")\n        >>> input_ids = inputs["input_ids"]\n        >>> labels = targets["input_ids"]\n        >>> outputs = model(input_ids=input_ids, labels=labels)\n\n        >>> # or use retriever separately\n        >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)\n        >>> # 1. Encode\n        >>> question_hidden_states = model.question_encoder(input_ids)[0]\n        >>> # 2. Retrieve\n        >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")\n        >>> doc_scores = torch.bmm(\n        ...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)\n        ... ).squeeze(1)\n        >>> # 3. Forward to generator\n        >>> outputs = model(\n        ...     context_input_ids=docs_dict["context_input_ids"],\n        ...     context_attention_mask=docs_dict["context_attention_mask"],\n        ...     doc_scores=doc_scores,\n        ...     decoder_input_ids=labels,\n        ... )\n\n        >>> # or directly generate\n        >>> generated = model.generate(\n        ...     context_input_ids=docs_dict["context_input_ids"],\n        ...     context_attention_mask=docs_dict["context_attention_mask"],\n        ...     doc_scores=doc_scores,\n        ... )\n        >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)\n        ```'
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_marginalize = do_marginalize if do_marginalize is not None else self.config.do_marginalize
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False
        outputs = self.rag(input_ids=input_ids, attention_mask=attention_mask, encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, context_input_ids=context_input_ids, context_attention_mask=context_attention_mask, doc_scores=doc_scores, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, output_retrieved=output_retrieved, n_docs=n_docs)
        loss = None
        logits = outputs.logits
        if labels is not None:
            assert decoder_input_ids is not None
            loss = self.get_nll(outputs.logits, outputs.doc_scores, labels, reduce_loss=reduce_loss, epsilon=self.config.label_smoothing, n_docs=n_docs)
        if do_marginalize:
            logits = self.marginalize(logits, outputs.doc_scores, n_docs)
        return RetrievAugLMMarginOutput(loss=loss, logits=logits, doc_scores=outputs.doc_scores, past_key_values=outputs.past_key_values, context_input_ids=outputs.context_input_ids, context_attention_mask=outputs.context_attention_mask, retrieved_doc_embeds=outputs.retrieved_doc_embeds, retrieved_doc_ids=outputs.retrieved_doc_ids, question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state, question_enc_hidden_states=outputs.question_enc_hidden_states, question_enc_attentions=outputs.question_enc_attentions, generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state, generator_enc_hidden_states=outputs.generator_enc_hidden_states, generator_enc_attentions=outputs.generator_enc_attentions, generator_dec_hidden_states=outputs.generator_dec_hidden_states, generator_dec_attentions=outputs.generator_dec_attentions, generator_cross_attentions=outputs.generator_cross_attentions)

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, context_input_ids: Optional[torch.LongTensor]=None, context_attention_mask: Optional[torch.LongTensor]=None, doc_scores: Optional[torch.FloatTensor]=None, n_docs: Optional[int]=None, generation_config: Optional[GenerationConfig]=None, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]]=None, logits_processor: Optional[LogitsProcessorList]=LogitsProcessorList(), stopping_criteria: Optional[StoppingCriteriaList]=StoppingCriteriaList(), **kwargs) -> torch.LongTensor:
        if False:
            for i in range(10):
                print('nop')
        "\n        Implements RAG token decoding.\n\n        Args:\n            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                The sequence used as a prompt for the generation. If `input_ids` is not passed, then\n                `context_input_ids` has to be provided.\n            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):\n                Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the\n                retriever.\n\n                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the\n                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].\n            context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):\n                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the\n                retriever.\n\n                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the\n                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].\n            doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):\n                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and\n                `question_encoder_last_hidden_state`.\n\n                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the\n                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].\n            n_docs (`int`, *optional*, defaults to `config.n_docs`)\n                Number of documents to retrieve and/or number of documents for which to generate an answer.\n            generation_config (`~generation.GenerationConfig`, *optional*):\n                The generation configuration to be used as base parametrization for the generation call. `**kwargs`\n                passed to generate matching the attributes of `generation_config` will override them. If\n                `generation_config` is not provided, the default will be used, which has the following loading\n                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model\n                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s\n                default values, whose documentation should be checked to parameterize generation.\n            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):\n                If provided, this function constraints the beam search to allowed tokens only at each step. If not\n                provided no constraint is applied. This function takes 2 arguments `inputs_ids` and the batch ID\n                `batch_id`. It has to return a list with the allowed tokens for the next generation step conditioned on\n                the previously generated tokens `inputs_ids` and the batch ID `batch_id`. This argument is useful for\n                constrained generation conditioned on the prefix, as described in [Autoregressive Entity\n                Retrieval](https://arxiv.org/abs/2010.00904).\n            logits_processor (`LogitsProcessorList`, *optional*):\n                Custom logits processors that complement the default logits processors built from arguments and a\n                model's config. If a logit processor is passed that is already created with the arguments or a model's\n                config an error is thrown.\n            stopping_criteria (`StoppingCriteriaList`, *optional*):\n                Custom stopping criteria that complement the default stopping criteria built from arguments and a\n                model's config. If a stopping criteria is passed that is already created with the arguments or a\n                model's config an error is thrown.\n            kwargs (`Dict[str, Any]`, *optional*):\n                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be\n                forwarded to the `forward` function of the model.\n\n        Return:\n            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated\n            sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches\n            finished early due to the `eos_token_id`.\n        "
        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            out = self.retriever(input_ids, question_hidden_states.cpu().detach().to(torch.float32).numpy(), prefix=self.generator.config.prefix, n_docs=n_docs, return_tensors='pt')
            (context_input_ids, context_attention_mask, retrieved_doc_embeds) = (out['context_input_ids'], out['context_attention_mask'], out['retrieved_doc_embeds'])
            retrieved_doc_embeds = retrieved_doc_embeds.to(question_hidden_states)
            context_input_ids = context_input_ids.to(input_ids)
            context_attention_mask = context_attention_mask.to(input_ids)
            doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
        assert context_input_ids.shape[0] % n_docs == 0, f' The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}.'
        batch_size = context_input_ids.shape[0] // n_docs
        encoder = self.rag.generator.get_encoder()
        encoder_outputs = encoder(input_ids=context_input_ids, attention_mask=context_attention_mask, return_dict=True)
        input_ids = torch.full((batch_size * generation_config.num_beams, 1), generation_config.decoder_start_token_id, dtype=torch.long, device=next(self.parameters()).device)
        input_ids_seq_length = input_ids.shape[-1]
        last_hidden_state = encoder_outputs['last_hidden_state']

        def extend_enc_output(tensor, num_beams=None):
            if False:
                return 10
            tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
            tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
            return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])
        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=generation_config.num_beams)
        encoder_outputs['last_hidden_state'] = extend_enc_output(last_hidden_state, num_beams=generation_config.num_beams)
        doc_scores = doc_scores.repeat_interleave(generation_config.num_beams, dim=0)
        model_kwargs['doc_scores'] = doc_scores
        model_kwargs['encoder_outputs'] = encoder_outputs
        model_kwargs['attention_mask'] = context_attention_mask
        model_kwargs['n_docs'] = n_docs
        pre_processor = self._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_seq_length, encoder_input_ids=context_input_ids, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, logits_processor=logits_processor)
        if generation_config.num_beams == 1:
            if generation_config.num_return_sequences > 1:
                raise ValueError(f'num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing greedy search.')
            return self.greedy_search(input_ids, logits_processor=pre_processor, max_length=generation_config.max_length, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, **model_kwargs)
        elif generation_config.num_beams > 1:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError('`num_return_sequences` has to be smaller or equal to `num_beams`.')
            beam_scorer = BeamSearchScorer(batch_size=batch_size, num_beams=generation_config.num_beams, device=self.device, length_penalty=generation_config.length_penalty, do_early_stopping=generation_config.early_stopping, num_beam_hyps_to_keep=generation_config.num_return_sequences, max_length=generation_config.max_length)
            return self.beam_search(input_ids, beam_scorer, logits_processor=pre_processor, max_length=generation_config.max_length, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, **model_kwargs)
        else:
            raise ValueError(f'`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {generation_config.num_beams}')

    def get_input_embeddings(self):
        if False:
            return 10
        return self.rag.generator.get_input_embeddings()

    def get_output_embeddings(self):
        if False:
            return 10
        return self.rag.generator.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        if False:
            while True:
                i = 10
        return self.rag.generator.set_output_embeddings(new_embeddings)

    def shift_tokens_right(self, input_ids, start_token_id=None):
        if False:
            i = 10
            return i + 15
        'Shift input ids one token to the right, and pad with start_token_id'
        if start_token_id is None:
            start_token_id = self.config.decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id
        return shifted_input_ids

    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        if False:
            print('Hello World!')
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        target = torch.cat([target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1)

        def _mask_pads(ll, smooth_obj):
            if False:
                for i in range(10):
                    print('nop')
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return (ll.squeeze(-1), smooth_obj.squeeze(-1))
        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)
        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()
        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)
        (ll, smooth_obj) = _mask_pads(ll, smooth_obj)
        ll = ll.sum(1)
        smooth_obj = smooth_obj.sum(1)
        nll_loss = -ll
        smooth_loss = -smooth_obj
        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss
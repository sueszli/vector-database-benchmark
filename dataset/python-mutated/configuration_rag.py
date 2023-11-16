""" RAG model configuration"""
from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings
RAG_CONFIG_DOC = '\n    [`RagConfig`] stores the configuration of a *RagModel*. Configuration objects inherit from [`PretrainedConfig`] and\n    can be used to control the model outputs. Read the documentation from [`PretrainedConfig`] for more information.\n\n    Args:\n        title_sep (`str`, *optional*, defaults to  `" / "`):\n            Separator inserted between the title and the text of the retrieved document when calling [`RagRetriever`].\n        doc_sep (`str`, *optional*, defaults to  `" // "`):\n            Separator inserted between the text of the retrieved document and the original input when calling\n            [`RagRetriever`].\n        n_docs (`int`, *optional*, defaults to 5):\n            Number of documents to retrieve.\n        max_combined_length (`int`, *optional*, defaults to 300):\n            Max length of contextualized input returned by [`~RagRetriever.__call__`].\n        retrieval_vector_size (`int`, *optional*, defaults to 768):\n            Dimensionality of the document embeddings indexed by [`RagRetriever`].\n        retrieval_batch_size (`int`, *optional*, defaults to 8):\n            Retrieval batch size, defined as the number of queries issues concurrently to the faiss index encapsulated\n            [`RagRetriever`].\n        dataset (`str`, *optional*, defaults to `"wiki_dpr"`):\n            A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and ids\n            using `datasets.list_datasets()`).\n        dataset_split (`str`, *optional*, defaults to `"train"`)\n            Which split of the `dataset` to load.\n        index_name (`str`, *optional*, defaults to `"compressed"`)\n            The index name of the index associated with the `dataset`. One can choose between `"legacy"`, `"exact"` and\n            `"compressed"`.\n        index_path (`str`, *optional*)\n            The path to the serialized faiss index on disk.\n        passages_path (`str`, *optional*):\n            A path to text passages compatible with the faiss index. Required if using\n            [`~models.rag.retrieval_rag.LegacyIndex`]\n        use_dummy_dataset (`bool`, *optional*, defaults to `False`)\n            Whether to load a "dummy" variant of the dataset specified by `dataset`.\n        label_smoothing (`float`, *optional*, defaults to 0.0):\n            Only relevant if `return_loss` is set to `True`. Controls the `epsilon` parameter value for label smoothing\n            in the loss calculation. If set to 0, no label smoothing is performed.\n        do_marginalize (`bool`, *optional*, defaults to `False`):\n            If `True`, the logits are marginalized over all documents by making use of\n            `torch.nn.functional.log_softmax`.\n        reduce_loss (`bool`, *optional*, defaults to `False`):\n            Whether or not to reduce the NLL loss using the `torch.Tensor.sum` operation.\n        do_deduplication (`bool`, *optional*, defaults to `True`):\n            Whether or not to deduplicate the generations from different context documents for a given input. Has to be\n            set to `False` if used while training with distributed backend.\n        exclude_bos_score (`bool`, *optional*, defaults to `False`):\n            Whether or not to disregard the BOS token when computing the loss.\n        output_retrieved(`bool`, *optional*, defaults to `False`):\n            If set to `True`, `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and\n            `context_attention_mask` are returned. See returned tensors for more detail.\n        use_cache (`bool`, *optional*, defaults to `True`):\n            Whether or not the model should return the last key/values attentions (not used by all models).\n        forced_eos_token_id (`int`, *optional*):\n            The id of the token to force as the last generated token when `max_length` is reached. Usually set to\n            `eos_token_id`.\n'

@add_start_docstrings(RAG_CONFIG_DOC)
class RagConfig(PretrainedConfig):
    model_type = 'rag'
    is_composition = True

    def __init__(self, vocab_size=None, is_encoder_decoder=True, prefix=None, bos_token_id=None, pad_token_id=None, eos_token_id=None, decoder_start_token_id=None, title_sep=' / ', doc_sep=' // ', n_docs=5, max_combined_length=300, retrieval_vector_size=768, retrieval_batch_size=8, dataset='wiki_dpr', dataset_split='train', index_name='compressed', index_path=None, passages_path=None, use_dummy_dataset=False, reduce_loss=False, label_smoothing=0.0, do_deduplication=True, exclude_bos_score=False, do_marginalize=False, output_retrieved=False, use_cache=True, forced_eos_token_id=None, **kwargs):
        if False:
            return 10
        super().__init__(bos_token_id=bos_token_id, pad_token_id=pad_token_id, eos_token_id=eos_token_id, decoder_start_token_id=decoder_start_token_id, forced_eos_token_id=forced_eos_token_id, is_encoder_decoder=is_encoder_decoder, prefix=prefix, vocab_size=vocab_size, **kwargs)
        assert 'question_encoder' in kwargs and 'generator' in kwargs, 'Config has to be initialized with question_encoder and generator config'
        question_encoder_config = kwargs.pop('question_encoder')
        question_encoder_model_type = question_encoder_config.pop('model_type')
        decoder_config = kwargs.pop('generator')
        decoder_model_type = decoder_config.pop('model_type')
        from ..auto.configuration_auto import AutoConfig
        self.question_encoder = AutoConfig.for_model(question_encoder_model_type, **question_encoder_config)
        self.generator = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.reduce_loss = reduce_loss
        self.label_smoothing = label_smoothing
        self.exclude_bos_score = exclude_bos_score
        self.do_marginalize = do_marginalize
        self.title_sep = title_sep
        self.doc_sep = doc_sep
        self.n_docs = n_docs
        self.max_combined_length = max_combined_length
        self.dataset = dataset
        self.dataset_split = dataset_split
        self.index_name = index_name
        self.retrieval_vector_size = retrieval_vector_size
        self.retrieval_batch_size = retrieval_batch_size
        self.passages_path = passages_path
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset
        self.output_retrieved = output_retrieved
        self.do_deduplication = do_deduplication
        self.use_cache = use_cache
        if self.forced_eos_token_id is None:
            self.forced_eos_token_id = getattr(self.generator, 'forced_eos_token_id', None)

    @classmethod
    def from_question_encoder_generator_configs(cls, question_encoder_config: PretrainedConfig, generator_config: PretrainedConfig, **kwargs) -> PretrainedConfig:
        if False:
            for i in range(10):
                print('nop')
        '\n        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and\n        decoder model configuration.\n\n        Returns:\n            [`EncoderDecoderConfig`]: An instance of a configuration object\n        '
        return cls(question_encoder=question_encoder_config.to_dict(), generator=generator_config.to_dict(), **kwargs)
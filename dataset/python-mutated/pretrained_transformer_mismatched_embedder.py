from typing import Optional, Dict, Any
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder, TokenEmbedder
from allennlp.nn import util

@TokenEmbedder.register('pretrained_transformer_mismatched')
class PretrainedTransformerMismatchedEmbedder(TokenEmbedder):
    """
    Use this embedder to embed wordpieces given by `PretrainedTransformerMismatchedIndexer`
    and to get word-level representations.

    Registered as a `TokenEmbedder` with name "pretrained_transformer_mismatched".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerMismatchedIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerMismatchedIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.
    override_weights_file: `Optional[str]`, optional (default = `None`)
        If set, this specifies a file from which to load alternate weights that override the
        weights from huggingface. The file is expected to contain a PyTorch `state_dict`, created
        with `torch.save()`.
    override_weights_strip_prefix: `Optional[str]`, optional (default = `None`)
        If set, strip the given prefix from the state dict when loading it.
    load_weights: `bool`, optional (default = `True`)
        Whether to load the pretrained weights. If you're loading your model/predictor from an AllenNLP archive
        it usually makes sense to set this to `False` (via the `overrides` parameter)
        to avoid unnecessarily caching and loading the original pretrained weights,
        since the archive will already contain all of the weights needed.
    gradient_checkpointing: `bool`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    tokenizer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/modeling_utils.py#L253)
        for `AutoModel.from_pretrained`.
    sub_token_mode: `Optional[str]`, optional (default= `avg`)
        If `sub_token_mode` is set to `first`, return first sub-token representation as word-level representation
        If `sub_token_mode` is set to `avg`, return average of all the sub-tokens representation as word-level representation
        If `sub_token_mode` is not specified it defaults to `avg`
        If invalid `sub_token_mode` is provided, throw `ConfigurationError`

    """

    def __init__(self, model_name: str, max_length: int=None, sub_module: str=None, train_parameters: bool=True, last_layer_only: bool=True, override_weights_file: Optional[str]=None, override_weights_strip_prefix: Optional[str]=None, load_weights: bool=True, gradient_checkpointing: Optional[bool]=None, tokenizer_kwargs: Optional[Dict[str, Any]]=None, transformer_kwargs: Optional[Dict[str, Any]]=None, sub_token_mode: Optional[str]='avg') -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._matched_embedder = PretrainedTransformerEmbedder(model_name, max_length=max_length, sub_module=sub_module, train_parameters=train_parameters, last_layer_only=last_layer_only, override_weights_file=override_weights_file, override_weights_strip_prefix=override_weights_strip_prefix, load_weights=load_weights, gradient_checkpointing=gradient_checkpointing, tokenizer_kwargs=tokenizer_kwargs, transformer_kwargs=transformer_kwargs)
        self.sub_token_mode = sub_token_mode

    def get_output_dim(self):
        if False:
            for i in range(10):
                print('nop')
        return self._matched_embedder.get_output_dim()

    def forward(self, token_ids: torch.LongTensor, mask: torch.BoolTensor, offsets: torch.LongTensor, wordpiece_mask: torch.BoolTensor, type_ids: Optional[torch.LongTensor]=None, segment_concat_mask: Optional[torch.BoolTensor]=None) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        # Parameters\n\n        token_ids: `torch.LongTensor`\n            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).\n        mask: `torch.BoolTensor`\n            Shape: [batch_size, num_orig_tokens].\n        offsets: `torch.LongTensor`\n            Shape: [batch_size, num_orig_tokens, 2].\n            Maps indices for the original tokens, i.e. those given as input to the indexer,\n            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`\n            corresponds to the original j-th token from the i-th batch.\n        wordpiece_mask: `torch.BoolTensor`\n            Shape: [batch_size, num_wordpieces].\n        type_ids: `Optional[torch.LongTensor]`\n            Shape: [batch_size, num_wordpieces].\n        segment_concat_mask: `Optional[torch.BoolTensor]`\n            See `PretrainedTransformerEmbedder`.\n\n        # Returns\n\n        `torch.Tensor`\n            Shape: [batch_size, num_orig_tokens, embedding_size].\n        '
        embeddings = self._matched_embedder(token_ids, wordpiece_mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask)
        (span_embeddings, span_mask) = util.batched_span_select(embeddings.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask
        if self.sub_token_mode == 'first':
            orig_embeddings = span_embeddings[:, :, 0, :]
        elif self.sub_token_mode == 'avg':
            span_embeddings_sum = span_embeddings.sum(2)
            span_embeddings_len = span_mask.sum(2)
            orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)
            orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0
        else:
            raise ConfigurationError(f"Do not recognise 'sub_token_mode' {self.sub_token_mode}")
        return orig_embeddings
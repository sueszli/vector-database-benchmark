""" Tokenization classes for LayoutXLM model."""
import os
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union
from ...tokenization_utils import AddedToken
from ...tokenization_utils_base import BatchEncoding, EncodedInput, PreTokenizedInput, TextInput, TextInputPair, TruncationStrategy
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, is_sentencepiece_available, logging
from ..xlm_roberta.tokenization_xlm_roberta_fast import PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES, PRETRAINED_VOCAB_FILES_MAP, VOCAB_FILES_NAMES
if is_sentencepiece_available():
    from .tokenization_layoutxlm import LayoutXLMTokenizer
else:
    LayoutXLMTokenizer = None
logger = logging.get_logger(__name__)
LAYOUTXLM_ENCODE_KWARGS_DOCSTRING = '\n            add_special_tokens (`bool`, *optional*, defaults to `True`):\n                Whether or not to encode the sequences with the special tokens relative to their model.\n            padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `False`):\n                Activates and controls padding. Accepts the following values:\n\n                - `True` or `\'longest\'`: Pad to the longest sequence in the batch (or no padding if only a single\n                  sequence if provided).\n                - `\'max_length\'`: Pad to a maximum length specified with the argument `max_length` or to the maximum\n                  acceptable input length for the model if that argument is not provided.\n                - `False` or `\'do_not_pad\'` (default): No padding (i.e., can output a batch with sequences of different\n                  lengths).\n            truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):\n                Activates and controls truncation. Accepts the following values:\n\n                - `True` or `\'longest_first\'`: Truncate to a maximum length specified with the argument `max_length` or\n                  to the maximum acceptable input length for the model if that argument is not provided. This will\n                  truncate token by token, removing a token from the longest sequence in the pair if a pair of\n                  sequences (or a batch of pairs) is provided.\n                - `\'only_first\'`: Truncate to a maximum length specified with the argument `max_length` or to the\n                  maximum acceptable input length for the model if that argument is not provided. This will only\n                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.\n                - `\'only_second\'`: Truncate to a maximum length specified with the argument `max_length` or to the\n                  maximum acceptable input length for the model if that argument is not provided. This will only\n                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.\n                - `False` or `\'do_not_truncate\'` (default): No truncation (i.e., can output batch with sequence lengths\n                  greater than the model maximum admissible input size).\n            max_length (`int`, *optional*):\n                Controls the maximum length to use by one of the truncation/padding parameters.\n\n                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length\n                is required by one of the truncation/padding parameters. If the model has no specific maximum input\n                length (like XLNet) truncation/padding to a maximum length will be deactivated.\n            stride (`int`, *optional*, defaults to 0):\n                If set to a number along with `max_length`, the overflowing tokens returned when\n                `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence\n                returned to provide some overlap between truncated and overflowing sequences. The value of this\n                argument defines the number of overlapping tokens.\n            pad_to_multiple_of (`int`, *optional*):\n                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable\n                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).\n            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):\n                If set, will return tensors instead of list of python integers. Acceptable values are:\n\n                - `\'tf\'`: Return TensorFlow `tf.constant` objects.\n                - `\'pt\'`: Return PyTorch `torch.Tensor` objects.\n                - `\'np\'`: Return Numpy `np.ndarray` objects.\n            return_token_type_ids (`bool`, *optional*):\n                Whether to return token type IDs. If left to the default, will return the token type IDs according to\n                the specific tokenizer\'s default, defined by the `return_outputs` attribute.\n\n                [What are token type IDs?](../glossary#token-type-ids)\n            return_attention_mask (`bool`, *optional*):\n                Whether to return the attention mask. If left to the default, will return the attention mask according\n                to the specific tokenizer\'s default, defined by the `return_outputs` attribute.\n\n                [What are attention masks?](../glossary#attention-mask)\n            return_overflowing_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch\n                of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead\n                of returning overflowing tokens.\n            return_special_tokens_mask (`bool`, *optional*, defaults to `False`):\n                Whether or not to return special tokens mask information.\n            return_offsets_mapping (`bool`, *optional*, defaults to `False`):\n                Whether or not to return `(char_start, char_end)` for each token.\n\n                This is only available on fast tokenizers inheriting from [`PreTrainedTokenizerFast`], if using\n                Python\'s tokenizer, this method will raise `NotImplementedError`.\n            return_length  (`bool`, *optional*, defaults to `False`):\n                Whether or not to return the lengths of the encoded inputs.\n            verbose (`bool`, *optional*, defaults to `True`):\n                Whether or not to print more information and warnings.\n            **kwargs: passed to the `self.tokenize()` method\n\n        Return:\n            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:\n\n            - **input_ids** -- List of token ids to be fed to a model.\n\n              [What are input IDs?](../glossary#input-ids)\n\n            - **bbox** -- List of bounding boxes to be fed to a model.\n\n            - **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or\n              if *"token_type_ids"* is in `self.model_input_names`).\n\n              [What are token type IDs?](../glossary#token-type-ids)\n\n            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when\n              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).\n\n              [What are attention masks?](../glossary#attention-mask)\n\n            - **labels** -- List of labels to be fed to a model. (when `word_labels` is specified).\n            - **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and\n              `return_overflowing_tokens=True`).\n            - **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and\n              `return_overflowing_tokens=True`).\n            - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying\n              regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).\n            - **length** -- The length of the inputs (when `return_length=True`).\n'

class LayoutXLMTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" LayoutXLM tokenizer (backed by HuggingFace's *tokenizers* library). Adapted from
    [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        cls_token_box (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`):
            The bounding box to use for the special [CLS] token.
        sep_token_box (`List[int]`, *optional*, defaults to `[1000, 1000, 1000, 1000]`):
            The bounding box to use for the special [SEP] token.
        pad_token_box (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`):
            The bounding box to use for the special [PAD] token.
        pad_token_label (`int`, *optional*, defaults to -100):
            The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch's
            CrossEntropyLoss.
        only_label_first_subword (`bool`, *optional*, defaults to `True`):
            Whether or not to only label the first subword, in case word labels are provided.
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ['input_ids', 'attention_mask']
    slow_tokenizer_class = LayoutXLMTokenizer

    def __init__(self, vocab_file=None, tokenizer_file=None, bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>', cls_token_box=[0, 0, 0, 0], sep_token_box=[1000, 1000, 1000, 1000], pad_token_box=[0, 0, 0, 0], pad_token_label=-100, only_label_first_subword=True, **kwargs):
        if False:
            i = 10
            return i + 15
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        super().__init__(vocab_file, tokenizer_file=tokenizer_file, bos_token=bos_token, eos_token=eos_token, sep_token=sep_token, cls_token=cls_token, unk_token=unk_token, pad_token=pad_token, mask_token=mask_token, cls_token_box=cls_token_box, sep_token_box=sep_token_box, pad_token_box=pad_token_box, pad_token_label=pad_token_label, only_label_first_subword=only_label_first_subword, **kwargs)
        self.vocab_file = vocab_file
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

    @property
    def can_save_slow_tokenizer(self) -> bool:
        if False:
            print('Hello World!')
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    @add_end_docstrings(LAYOUTXLM_ENCODE_KWARGS_DOCSTRING)
    def __call__(self, text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]], text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]]=None, boxes: Union[List[List[int]], List[List[List[int]]]]=None, word_labels: Optional[Union[List[int], List[List[int]]]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[Union[str, TensorType]]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        if False:
            print('Hello World!')
        '\n        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of\n        sequences with word-level normalized bounding boxes and optional labels.\n\n        Args:\n            text (`str`, `List[str]`, `List[List[str]]`):\n                The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings\n                (words of a single example or questions of a batch of examples) or a list of list of strings (batch of\n                words).\n            text_pair (`List[str]`, `List[List[str]]`):\n                The sequence or batch of sequences to be encoded. Each sequence should be a list of strings\n                (pretokenized string).\n            boxes (`List[List[int]]`, `List[List[List[int]]]`):\n                Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.\n            word_labels (`List[int]`, `List[List[int]]`, *optional*):\n                Word-level integer labels (for token classification tasks such as FUNSD, CORD).\n        '

        def _is_valid_text_input(t):
            if False:
                i = 10
                return i + 15
            if isinstance(t, str):
                return True
            elif isinstance(t, (list, tuple)):
                if len(t) == 0:
                    return True
                elif isinstance(t[0], str):
                    return True
                elif isinstance(t[0], (list, tuple)):
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False
        if text_pair is not None:
            if not _is_valid_text_input(text):
                raise ValueError('text input must of type `str` (single example) or `List[str]` (batch of examples). ')
            if not isinstance(text_pair, (list, tuple)):
                raise ValueError('words must of type `List[str]` (single pretokenized example), or `List[List[str]]` (batch of pretokenized examples).')
        elif not isinstance(text, (list, tuple)):
            raise ValueError('Words must of type `List[str]` (single pretokenized example), or `List[List[str]]` (batch of pretokenized examples).')
        if text_pair is not None:
            is_batched = isinstance(text, (list, tuple))
        else:
            is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
        words = text if text_pair is None else text_pair
        if boxes is None:
            raise ValueError('You must provide corresponding bounding boxes')
        if is_batched:
            if len(words) != len(boxes):
                raise ValueError('You must provide words and boxes for an equal amount of examples')
            for (words_example, boxes_example) in zip(words, boxes):
                if len(words_example) != len(boxes_example):
                    raise ValueError('You must provide as many words as there are bounding boxes')
        elif len(words) != len(boxes):
            raise ValueError('You must provide as many words as there are bounding boxes')
        if is_batched:
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(f'batch length of `text`: {len(text)} does not match batch length of `text_pair`: {len(text_pair)}.')
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            is_pair = bool(text_pair is not None)
            return self.batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs, is_pair=is_pair, boxes=boxes, word_labels=word_labels, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)
        else:
            return self.encode_plus(text=text, text_pair=text_pair, boxes=boxes, word_labels=word_labels, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)

    def tokenize(self, text: str, pair: Optional[str]=None, add_special_tokens: bool=False, **kwargs) -> List[str]:
        if False:
            return 10
        batched_input = [(text, pair)] if pair else [text]
        encodings = self._tokenizer.encode_batch(batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs)
        return encodings[0].tokens

    def _batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair], List[PreTokenizedInput]], is_pair: bool=None, boxes: Optional[List[List[List[int]]]]=None, word_labels: Optional[List[List[int]]]=None, add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[str]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        if False:
            while True:
                i = 10
        if not isinstance(batch_text_or_text_pairs, list):
            raise TypeError(f'batch_text_or_text_pairs has to be a list (got {type(batch_text_or_text_pairs)})')
        self.set_truncation_and_padding(padding_strategy=padding_strategy, truncation_strategy=truncation_strategy, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of)
        if is_pair:
            batch_text_or_text_pairs = [(text.split(), text_pair) for (text, text_pair) in batch_text_or_text_pairs]
        encodings = self._tokenizer.encode_batch(batch_text_or_text_pairs, add_special_tokens=add_special_tokens, is_pretokenized=True)
        tokens_and_encodings = [self._convert_encoding(encoding=encoding, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=True if word_labels is not None else return_offsets_mapping, return_length=return_length, verbose=verbose) for encoding in encodings]
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0].keys():
            stack = [e for (item, _) in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
        sanitized_encodings = [e for (_, item) in tokens_and_encodings for e in item]
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for (i, (toks, _)) in enumerate(tokens_and_encodings):
                overflow_to_sample_mapping += [i] * len(toks['input_ids'])
            sanitized_tokens['overflow_to_sample_mapping'] = overflow_to_sample_mapping
        for input_ids in sanitized_tokens['input_ids']:
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
        token_boxes = []
        for batch_index in range(len(sanitized_tokens['input_ids'])):
            if return_overflowing_tokens:
                original_index = sanitized_tokens['overflow_to_sample_mapping'][batch_index]
            else:
                original_index = batch_index
            token_boxes_example = []
            for (id, sequence_id, word_id) in zip(sanitized_tokens['input_ids'][batch_index], sanitized_encodings[batch_index].sequence_ids, sanitized_encodings[batch_index].word_ids):
                if word_id is not None:
                    if is_pair and sequence_id == 0:
                        token_boxes_example.append(self.pad_token_box)
                    else:
                        token_boxes_example.append(boxes[original_index][word_id])
                elif id == self.cls_token_id:
                    token_boxes_example.append(self.cls_token_box)
                elif id == self.sep_token_id:
                    token_boxes_example.append(self.sep_token_box)
                elif id == self.pad_token_id:
                    token_boxes_example.append(self.pad_token_box)
                else:
                    raise ValueError('Id not recognized')
            token_boxes.append(token_boxes_example)
        sanitized_tokens['bbox'] = token_boxes
        if word_labels is not None:
            labels = []
            for batch_index in range(len(sanitized_tokens['input_ids'])):
                if return_overflowing_tokens:
                    original_index = sanitized_tokens['overflow_to_sample_mapping'][batch_index]
                else:
                    original_index = batch_index
                labels_example = []
                for (id, offset, word_id) in zip(sanitized_tokens['input_ids'][batch_index], sanitized_tokens['offset_mapping'][batch_index], sanitized_encodings[batch_index].word_ids):
                    if word_id is not None:
                        if self.only_label_first_subword:
                            if offset[0] == 0:
                                labels_example.append(word_labels[original_index][word_id])
                            else:
                                labels_example.append(self.pad_token_label)
                        else:
                            labels_example.append(word_labels[original_index][word_id])
                    else:
                        labels_example.append(self.pad_token_label)
                labels.append(labels_example)
            sanitized_tokens['labels'] = labels
            if not return_offsets_mapping:
                del sanitized_tokens['offset_mapping']
        return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

    def _encode_plus(self, text: Union[TextInput, PreTokenizedInput], text_pair: Optional[PreTokenizedInput]=None, boxes: Optional[List[List[int]]]=None, word_labels: Optional[List[int]]=None, add_special_tokens: bool=True, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, truncation_strategy: TruncationStrategy=TruncationStrategy.DO_NOT_TRUNCATE, max_length: Optional[int]=None, stride: int=0, pad_to_multiple_of: Optional[int]=None, return_tensors: Optional[bool]=None, return_token_type_ids: Optional[bool]=None, return_attention_mask: Optional[bool]=None, return_overflowing_tokens: bool=False, return_special_tokens_mask: bool=False, return_offsets_mapping: bool=False, return_length: bool=False, verbose: bool=True, **kwargs) -> BatchEncoding:
        if False:
            for i in range(10):
                print('nop')
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_boxes = [boxes]
        batched_word_labels = [word_labels] if word_labels is not None else None
        batched_output = self._batch_encode_plus(batched_input, is_pair=bool(text_pair is not None), boxes=batched_boxes, word_labels=batched_word_labels, add_special_tokens=add_special_tokens, padding_strategy=padding_strategy, truncation_strategy=truncation_strategy, max_length=max_length, stride=stride, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask, return_overflowing_tokens=return_overflowing_tokens, return_special_tokens_mask=return_special_tokens_mask, return_offsets_mapping=return_offsets_mapping, return_length=return_length, verbose=verbose, **kwargs)
        if return_tensors is None and (not return_overflowing_tokens):
            batched_output = BatchEncoding({key: value[0] if len(value) > 0 and isinstance(value[0], list) else value for (key, value) in batched_output.items()}, batched_output.encodings)
        self._eventual_warn_about_too_long_sequence(batched_output['input_ids'], max_length, verbose)
        return batched_output

    def _pad(self, encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding], max_length: Optional[int]=None, padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None) -> dict:
        if False:
            print('Hello World!')
        "\n        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)\n\n        Args:\n            encoded_inputs:\n                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).\n            max_length: maximum length of the returned list and optionally padding length (see below).\n                Will truncate by taking into account the special tokens.\n            padding_strategy: PaddingStrategy to use for padding.\n\n                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch\n                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)\n                - PaddingStrategy.DO_NOT_PAD: Do not pad\n                The tokenizer padding sides are defined in self.padding_side:\n\n                    - 'left': pads on the left of the sequences\n                    - 'right': pads on the right of the sequences\n            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.\n                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability\n                `>= 7.5` (Volta).\n            return_attention_mask:\n                (optional) Set to False to avoid returning attention mask (default: set to model specifics)\n        "
        if return_attention_mask is None:
            return_attention_mask = 'attention_mask' in self.model_input_names
        required_input = encoded_inputs[self.model_input_names[0]]
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
        if return_attention_mask and 'attention_mask' not in encoded_inputs:
            encoded_inputs['attention_mask'] = [1] * len(required_input)
        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs['attention_mask'] = encoded_inputs['attention_mask'] + [0] * difference
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = encoded_inputs['token_type_ids'] + [self.pad_token_type_id] * difference
                if 'bbox' in encoded_inputs:
                    encoded_inputs['bbox'] = encoded_inputs['bbox'] + [self.pad_token_box] * difference
                if 'labels' in encoded_inputs:
                    encoded_inputs['labels'] = encoded_inputs['labels'] + [self.pad_token_label] * difference
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = encoded_inputs['special_tokens_mask'] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs['attention_mask'] = [0] * difference + encoded_inputs['attention_mask']
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = [self.pad_token_type_id] * difference + encoded_inputs['token_type_ids']
                if 'bbox' in encoded_inputs:
                    encoded_inputs['bbox'] = [self.pad_token_box] * difference + encoded_inputs['bbox']
                if 'labels' in encoded_inputs:
                    encoded_inputs['labels'] = [self.pad_token_label] * difference + encoded_inputs['labels']
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = [1] * difference + encoded_inputs['special_tokens_mask']
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError('Invalid padding strategy:' + str(self.padding_side))
        return encoded_inputs

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. An XLM-RoBERTa sequence has the following format:\n\n        - single sequence: `<s> X </s>`\n        - pair of sequences: `<s> A </s></s> B </s>`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does\n        not make use of token type ids, therefore a list of zeros is returned.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of zeros.\n\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            i = 10
            return i + 15
        if not self.can_save_slow_tokenizer:
            raise ValueError('Your fast tokenizer does not have the necessary information to save the vocabulary for a slow tokenizer.')
        if not os.path.isdir(save_directory):
            logger.error(f'Vocabulary path ({save_directory}) should be a directory.')
            return
        out_vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        return (out_vocab_file,)
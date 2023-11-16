"""Fast Tokenization classes for REALM."""
import json
from typing import List, Optional, Tuple
from tokenizers import normalizers
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, logging
from .tokenization_realm import RealmTokenizer
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'google/realm-cc-news-pretrained-embedder': 'https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/vocab.txt', 'google/realm-cc-news-pretrained-encoder': 'https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/vocab.txt', 'google/realm-cc-news-pretrained-scorer': 'https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/vocab.txt', 'google/realm-cc-news-pretrained-openqa': 'https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/vocab.txt', 'google/realm-orqa-nq-openqa': 'https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/vocab.txt', 'google/realm-orqa-nq-reader': 'https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/vocab.txt', 'google/realm-orqa-wq-openqa': 'https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/vocab.txt', 'google/realm-orqa-wq-reader': 'https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/vocab.txt'}, 'tokenizer_file': {'google/realm-cc-news-pretrained-embedder': 'https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/tokenizer.jsont', 'google/realm-cc-news-pretrained-encoder': 'https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/tokenizer.json', 'google/realm-cc-news-pretrained-scorer': 'https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/tokenizer.json', 'google/realm-cc-news-pretrained-openqa': 'https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/tokenizer.json', 'google/realm-orqa-nq-openqa': 'https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/tokenizer.json', 'google/realm-orqa-nq-reader': 'https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/tokenizer.json', 'google/realm-orqa-wq-openqa': 'https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/tokenizer.json', 'google/realm-orqa-wq-reader': 'https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/tokenizer.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'google/realm-cc-news-pretrained-embedder': 512, 'google/realm-cc-news-pretrained-encoder': 512, 'google/realm-cc-news-pretrained-scorer': 512, 'google/realm-cc-news-pretrained-openqa': 512, 'google/realm-orqa-nq-openqa': 512, 'google/realm-orqa-nq-reader': 512, 'google/realm-orqa-wq-openqa': 512, 'google/realm-orqa-wq-reader': 512}
PRETRAINED_INIT_CONFIGURATION = {'google/realm-cc-news-pretrained-embedder': {'do_lower_case': True}, 'google/realm-cc-news-pretrained-encoder': {'do_lower_case': True}, 'google/realm-cc-news-pretrained-scorer': {'do_lower_case': True}, 'google/realm-cc-news-pretrained-openqa': {'do_lower_case': True}, 'google/realm-orqa-nq-openqa': {'do_lower_case': True}, 'google/realm-orqa-nq-reader': {'do_lower_case': True}, 'google/realm-orqa-wq-openqa': {'do_lower_case': True}, 'google/realm-orqa-wq-reader': {'do_lower_case': True}}

class RealmTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" REALM tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    [`RealmTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = RealmTokenizer

    def __init__(self, vocab_file=None, tokenizer_file=None, do_lower_case=True, unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]', tokenize_chinese_chars=True, strip_accents=None, **kwargs):
        if False:
            return 10
        super().__init__(vocab_file, tokenizer_file=tokenizer_file, do_lower_case=do_lower_case, unk_token=unk_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, tokenize_chinese_chars=tokenize_chinese_chars, strip_accents=strip_accents, **kwargs)
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if normalizer_state.get('lowercase', do_lower_case) != do_lower_case or normalizer_state.get('strip_accents', strip_accents) != strip_accents or normalizer_state.get('handle_chinese_chars', tokenize_chinese_chars) != tokenize_chinese_chars:
            normalizer_class = getattr(normalizers, normalizer_state.pop('type'))
            normalizer_state['lowercase'] = do_lower_case
            normalizer_state['strip_accents'] = strip_accents
            normalizer_state['handle_chinese_chars'] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)
        self.do_lower_case = do_lower_case

    def batch_encode_candidates(self, text, **kwargs):
        if False:
            return 10
        '\n        Encode a batch of text or text pair. This method is similar to regular __call__ method but has the following\n        differences:\n\n            1. Handle additional num_candidate axis. (batch_size, num_candidates, text)\n            2. Always pad the sequences to *max_length*.\n            3. Must specify *max_length* in order to stack packs of candidates into a batch.\n\n            - single sequence: `[CLS] X [SEP]`\n            - pair of sequences: `[CLS] A [SEP] B [SEP]`\n\n        Args:\n            text (`List[List[str]]`):\n                The batch of sequences to be encoded. Each sequence must be in this format: (batch_size,\n                num_candidates, text).\n            text_pair (`List[List[str]]`, *optional*):\n                The batch of sequences to be encoded. Each sequence must be in this format: (batch_size,\n                num_candidates, text).\n            **kwargs:\n                Keyword arguments of the __call__ method.\n\n        Returns:\n            [`BatchEncoding`]: Encoded text or text pair.\n\n        Example:\n\n        ```python\n        >>> from transformers import RealmTokenizerFast\n\n        >>> # batch_size = 2, num_candidates = 2\n        >>> text = [["Hello world!", "Nice to meet you!"], ["The cute cat.", "The adorable dog."]]\n\n        >>> tokenizer = RealmTokenizerFast.from_pretrained("google/realm-cc-news-pretrained-encoder")\n        >>> tokenized_text = tokenizer.batch_encode_candidates(text, max_length=10, return_tensors="pt")\n        ```'
        kwargs['padding'] = PaddingStrategy.MAX_LENGTH
        batch_text = text
        batch_text_pair = kwargs.pop('text_pair', None)
        return_tensors = kwargs.pop('return_tensors', None)
        output_data = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
        for (idx, candidate_text) in enumerate(batch_text):
            if batch_text_pair is not None:
                candidate_text_pair = batch_text_pair[idx]
            else:
                candidate_text_pair = None
            encoded_candidates = super().__call__(candidate_text, candidate_text_pair, return_tensors=None, **kwargs)
            encoded_input_ids = encoded_candidates.get('input_ids')
            encoded_attention_mask = encoded_candidates.get('attention_mask')
            encoded_token_type_ids = encoded_candidates.get('token_type_ids')
            if encoded_input_ids is not None:
                output_data['input_ids'].append(encoded_input_ids)
            if encoded_attention_mask is not None:
                output_data['attention_mask'].append(encoded_attention_mask)
            if encoded_token_type_ids is not None:
                output_data['token_type_ids'].append(encoded_token_type_ids)
        output_data = {key: item for (key, item) in output_data.items() if len(item) != 0}
        return BatchEncoding(output_data, tensor_type=return_tensors)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if False:
            i = 10
            return i + 15
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. A REALM sequence has the following format:\n\n        - single sequence: `[CLS] X [SEP]`\n        - pair of sequences: `[CLS] A [SEP] B [SEP]`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]
        return output

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A REALM sequence\n        pair mask has the following format:\n\n        ```\n        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1\n        | first sequence    | second sequence |\n        ```\n\n        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            for i in range(10):
                print('nop')
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
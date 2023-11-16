"""Fast Tokenization classes for MPNet."""
import json
from typing import List, Optional, Tuple
from tokenizers import normalizers
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_mpnet import MPNetTokenizer
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'microsoft/mpnet-base': 'https://huggingface.co/microsoft/mpnet-base/resolve/main/vocab.txt'}, 'tokenizer_file': {'microsoft/mpnet-base': 'https://huggingface.co/microsoft/mpnet-base/resolve/main/tokenizer.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'microsoft/mpnet-base': 512}
PRETRAINED_INIT_CONFIGURATION = {'microsoft/mpnet-base': {'do_lower_case': True}}

class MPNetTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" MPNet tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
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
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = MPNetTokenizer
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, vocab_file=None, tokenizer_file=None, do_lower_case=True, bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='[UNK]', pad_token='<pad>', mask_token='<mask>', tokenize_chinese_chars=True, strip_accents=None, **kwargs):
        if False:
            print('Hello World!')
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        super().__init__(vocab_file, tokenizer_file=tokenizer_file, do_lower_case=do_lower_case, bos_token=bos_token, eos_token=eos_token, sep_token=sep_token, cls_token=cls_token, unk_token=unk_token, pad_token=pad_token, mask_token=mask_token, tokenize_chinese_chars=tokenize_chinese_chars, strip_accents=strip_accents, **kwargs)
        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if pre_tok_state.get('lowercase', do_lower_case) != do_lower_case or pre_tok_state.get('strip_accents', strip_accents) != strip_accents:
            pre_tok_class = getattr(normalizers, pre_tok_state.pop('type'))
            pre_tok_state['lowercase'] = do_lower_case
            pre_tok_state['strip_accents'] = strip_accents
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)
        self.do_lower_case = do_lower_case

    @property
    def mask_token(self) -> str:
        if False:
            return 10
        '\n        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not\n        having been set.\n\n        MPNet tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily\n        comprise the space before the *<mask>*.\n        '
        if self._mask_token is None:
            if self.verbose:
                logger.error('Using mask_token, but it is not set yet.')
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        if False:
            i = 10
            return i + 15
        '\n        Overriding the default behavior of the mask token to have it eat the space before it.\n\n        This is needed to preserve backward compatibility with all the previously used models based on MPNet.\n        '
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if False:
            while True:
                i = 10
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. MPNet does not\n        make use of token type ids, therefore a list of zeros is returned\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of ids.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs\n\n        Returns:\n            `List[int]`: List of zeros.\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            return 10
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
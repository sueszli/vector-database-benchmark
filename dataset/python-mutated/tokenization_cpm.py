"""Tokenization classes."""
import os
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import SPIECE_UNDERLINE, logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'spiece.model'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'TsinghuaAI/CPM-Generate': 'https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/spiece.model'}}

class CpmTokenizer(PreTrainedTokenizer):
    """Runs pre-tokenization with Jieba segmentation tool. It is used in CPM models."""
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP

    def __init__(self, vocab_file, do_lower_case=False, remove_space=True, keep_accents=False, bos_token='<s>', eos_token='</s>', unk_token='<unk>', sep_token='<sep>', pad_token='<pad>', cls_token='<cls>', mask_token='<mask>', additional_special_tokens=['<eop>', '<eod>'], sp_model_kwargs: Optional[Dict[str, Any]]=None, **kwargs) -> None:
        if False:
            return 10
        '\n        Construct a CPM tokenizer. Based on [Jieba](https://pypi.org/project/jieba/) and\n        [SentencePiece](https://github.com/google/sentencepiece).\n\n        This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should\n        refer to this superclass for more information regarding those methods.\n\n        Args:\n            vocab_file (`str`):\n                [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that\n                contains the vocabulary necessary to instantiate a tokenizer.\n            do_lower_case (`bool`, *optional*, defaults to `True`):\n                Whether to lowercase the input when tokenizing.\n            remove_space (`bool`, *optional*, defaults to `True`):\n                Whether to strip the text when tokenizing (removing excess spaces before and after the string).\n            keep_accents (`bool`, *optional*, defaults to `False`):\n                Whether to keep accents when tokenizing.\n            bos_token (`str`, *optional*, defaults to `"<s>"`):\n                The beginning of sequence token that was used during pretraining. Can be used a sequence classifier\n                token.\n\n                <Tip>\n\n                When building a sequence using special tokens, this is not the token that is used for the beginning of\n                sequence. The token used is the `cls_token`.\n\n                </Tip>\n\n            eos_token (`str`, *optional*, defaults to `"</s>"`):\n                The end of sequence token.\n\n                <Tip>\n\n                When building a sequence using special tokens, this is not the token that is used for the end of\n                sequence. The token used is the `sep_token`.\n\n                </Tip>\n\n            unk_token (`str`, *optional*, defaults to `"<unk>"`):\n                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be\n                this token instead.\n            sep_token (`str`, *optional*, defaults to `"<sep>"`):\n                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences\n                for sequence classification or for a text and a question for question answering. It is also used as the\n                last token of a sequence built with special tokens.\n            pad_token (`str`, *optional*, defaults to `"<pad>"`):\n                The token used for padding, for example when batching sequences of different lengths.\n            cls_token (`str`, *optional*, defaults to `"<cls>"`):\n                The classifier token which is used when doing sequence classification (classification of the whole\n                sequence instead of per-token classification). It is the first token of the sequence when built with\n                special tokens.\n            mask_token (`str`, *optional*, defaults to `"<mask>"`):\n                The token used for masking values. This is the token used when training this model with masked language\n                modeling. This is the token which the model will try to predict.\n            additional_special_tokens (`List[str]`, *optional*, defaults to `["<eop>", "<eod>"]`):\n                Additional special tokens used by the tokenizer.\n\n        Attributes:\n            sp_model (`SentencePieceProcessor`):\n                The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).\n        '
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        try:
            import jieba
        except ModuleNotFoundError as error:
            raise error.__class__('You need to install jieba to use CpmTokenizer or CpmTokenizerFast. See https://pypi.org/project/jieba/ for installation.')
        self.jieba = jieba
        self.translator = str.maketrans(' \n', '▂▃')
        super().__init__(do_lower_case=do_lower_case, remove_space=remove_space, keep_accents=keep_accents, bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, additional_special_tokens=additional_special_tokens, sp_model_kwargs=self.sp_model_kwargs, **kwargs)
        self._pad_token_type_id = 3

    @property
    def vocab_size(self):
        if False:
            i = 10
            return i + 15
        return len(self.sp_model)

    def get_vocab(self):
        if False:
            i = 10
            return i + 15
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        state = self.__dict__.copy()
        state['sp_model'] = None
        return state

    def __setstate__(self, d):
        if False:
            print('Hello World!')
        self.__dict__ = d
        if not hasattr(self, 'sp_model_kwargs'):
            self.sp_model_kwargs = {}
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs):
        if False:
            while True:
                i = 10
        if self.remove_space:
            outputs = ' '.join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace('``', '"').replace("''", '"')
        if not self.keep_accents:
            outputs = unicodedata.normalize('NFKD', outputs)
            outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()
        return outputs

    def _tokenize(self, text: str) -> List[str]:
        if False:
            while True:
                i = 10
        'Tokenize a string.'
        text = self.preprocess_text(text)
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(',') and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ''))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)
        return new_pieces

    def _convert_token_to_id(self, token):
        if False:
            i = 10
            return i + 15
        'Converts a token (str) in an id using the vocab.'
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        if False:
            return 10
        'Converts an index (integer) in a token (str) using the vocab.'
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        if False:
            return 10
        'Converts a sequence of tokens (strings for sub-words) in a single string.'
        out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. An XLNet sequence has the following format:\n\n        - single sequence: `X <sep> <cls>`\n        - pair of sequences: `A <sep> B <sep> <cls>`\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return token_ids_0 + sep + cls
        return token_ids_0 + sep + token_ids_1 + sep + cls

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None, already_has_special_tokens: bool=False) -> List[int]:
        if False:
            print('Hello World!')
        '\n        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding\n        special tokens using the tokenizer `prepare_for_model` method.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n            already_has_special_tokens (`bool`, *optional*, defaults to `False`):\n                Whether or not the token list is already formatted with special tokens for the model.\n\n        Returns:\n            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.\n        '
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
        if token_ids_1 is not None:
            return [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1, 1]
        return [0] * len(token_ids_0) + [1, 1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            return 10
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet\n        sequence pair mask has the following format:\n\n        ```\n        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1\n        | first sequence    | second sequence |\n        ```\n\n        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).\n        '
        sep = [self.sep_token_id]
        cls_segment_id = [2]
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0] + cls_segment_id
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            while True:
                i = 10
        if not os.path.isdir(save_directory):
            logger.error(f'Vocabulary path ({save_directory}) should be a directory')
            return
        out_vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, 'wb') as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        return (out_vocab_file,)

    def _decode(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        text = super()._decode(*args, **kwargs)
        text = text.replace(' ', '').replace('▂', ' ').replace('▃', '\n')
        return text
"""Tokenization classes for OpenAI GPT."""
from typing import List, Optional, Tuple
from tokenizers import pre_tokenizers
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_clip import CLIPTokenizer
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'merges_file': 'merges.txt', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'openai/clip-vit-base-patch32': 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json'}, 'merges_file': {'openai/clip-vit-base-patch32': 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt'}, 'tokenizer_file': {'openai/clip-vit-base-patch32': 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'openai/clip-vit-base-patch32': 77}

class CLIPTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" CLIP tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ['input_ids', 'attention_mask']
    slow_tokenizer_class = CLIPTokenizer

    def __init__(self, vocab_file=None, merges_file=None, tokenizer_file=None, unk_token='<|endoftext|>', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|endoftext|>', **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(vocab_file, merges_file, tokenizer_file=tokenizer_file, unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, **kwargs)
        if not isinstance(self.backend_tokenizer.pre_tokenizer, pre_tokenizers.Sequence):
            raise ValueError('The `backend_tokenizer` provided does not match the expected format. The CLIP tokenizer has been heavily modified from transformers version 4.17.0. You need to convert the tokenizer you are using to be compatible with this version.The easiest way to do so is `CLIPTokenizerFast.from_pretrained("path_to_local_folder_or_hub_repo, from_slow=True)`. If you want to use your existing tokenizer, you will have to revert to a version prior to 4.17.0 of transformers.')
        self._wrap_decode_method_backend_tokenizer()

    def _wrap_decode_method_backend_tokenizer(self):
        if False:
            print('Hello World!')
        orig_decode_method = self.backend_tokenizer.decode

        def new_decode_method(*args, **kwargs):
            if False:
                return 10
            text = orig_decode_method(*args, **kwargs)
            text = text.replace(self.backend_tokenizer.model.end_of_word_suffix, ' ').strip()
            return text
        self.backend_tokenizer.decode = new_decode_method

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            while True:
                i = 10
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. A CLIP sequence has the following format:\n\n        - single sequence: `<|startoftext|> X <|endoftext|>`\n\n        Pairs of sequences are not the expected use case, but they will be handled without a separator.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        bos_token = [self.bos_token_id]
        eos_token = [self.eos_token_id]
        if token_ids_1 is None:
            return bos_token + token_ids_0 + eos_token
        return bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            i = 10
            return i + 15
        '\n        Create a mask from the two sequences passed. CLIP does not make use of token type ids, therefore a list of\n        zeros is returned.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of zeros.\n        '
        bos_token = [self.bos_token_id]
        eos_token = [self.eos_token_id]
        if token_ids_1 is None:
            return len(bos_token + token_ids_0 + eos_token) * [0]
        return len(bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            i = 10
            return i + 15
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
"""Fast tokenization class for BlenderbotSmall."""
from typing import List, Optional
from tokenizers import ByteLevelBPETokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_blenderbot_small import BlenderbotSmallTokenizer
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'merges_file': 'merges.txt', 'tokenizer_config_file': 'tokenizer_config.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'facebook/blenderbot_small-90M': 'https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/vocab.json'}, 'merges_file': {'facebook/blenderbot_small-90M': 'https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/merges.txt'}, 'tokenizer_config_file': {'facebook/blenderbot_small-90M': 'https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/tokenizer_config.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'facebook/blenderbot_small-90M': 512}

class BlenderbotSmallTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" BlenderbotSmall tokenizer (backed by HuggingFace's *tokenizers* library).

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = BlenderbotSmallTokenizer

    def __init__(self, vocab_file=None, merges_file=None, unk_token='<|endoftext|>', bos_token='<|endoftext|>', eos_token='<|endoftext|>', add_prefix_space=False, trim_offsets=True, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(ByteLevelBPETokenizer(vocab=vocab_file, merges=merges_file, add_prefix_space=add_prefix_space, trim_offsets=trim_offsets), bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)
        self.add_prefix_space = add_prefix_space

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if False:
            return 10
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            return 10
        '\n        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BlenderbotSmall\n        does not make use of token type ids, therefore a list of zeros is returned.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: List of zeros.\n        '
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def default_chat_template(self):
        if False:
            while True:
                i = 10
        '\n        A very simple chat template that just adds whitespace between messages.\n        '
        logger.warning_once(f'\nNo chat template is defined for this tokenizer - using the default template for the {self.__class__.__name__} class. If the default is not appropriate for your model, please set `tokenizer.chat_template` to an appropriate template. See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n')
        return "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
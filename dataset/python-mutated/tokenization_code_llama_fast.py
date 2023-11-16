import os
from shutil import copyfile
from typing import List, Optional, Tuple
from tokenizers import normalizers, processors
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
from ...utils.versions import require_version
require_version('tokenizers>=0.13.3')
if is_sentencepiece_available():
    from .tokenization_code_llama import CodeLlamaTokenizer
else:
    CodeLlamaTokenizer = None
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'tokenizer.model', 'tokenizer_file': 'tokenizer.json'}
SPIECE_UNDERLINE = '▁'
(B_INST, E_INST) = ('[INST]', '[/INST]')
(B_SYS, E_SYS) = ('<<SYS>>\n', '\n<</SYS>>\n\n')
DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

class CodeLlamaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and no normalization.

    ```python
    >>> from transformers import CodeLlamaTokenizerFast

    >>> tokenizer = CodeLlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    >>> tokenizer.encode("Hello this is a test")
    [1, 15043, 445, 338, 263, 1243]
    ```

    If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
    call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
    values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
    [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.


    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods. The default configuration match that of
    [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json)
    which supports prompt infilling.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        clean_up_tokenization_spaces (`str`, *optional*, defaults to `False`):
            Wether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra
            spaces.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        prefix_token (`str`, *optional*, defaults to `"▁<PRE>"`):
            Prefix token used for infilling.
        middle_token (`str`, *optional*, defaults to `"▁<MID>"`):
            Middle token used for infilling.
        suffix_token (`str`, *optional*, defaults to `"▁<SUF>"`):
            Suffix token used for infilling.
        eot_token (`str`, *optional*, defaults to `"▁<EOT>"`):
            End of text token used for infilling.
        fill_token (`str`, *optional*, defaults to `"<FILL_ME>"`):
            The token used to split the input between the prefix and suffix.
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether to add a beginning of sequence token at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether to add an end of sequence token at the end of sequences.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Llama should be used.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = CodeLlamaTokenizer
    padding_side = 'left'
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, vocab_file=None, tokenizer_file=None, clean_up_tokenization_spaces=False, unk_token='<unk>', bos_token='<s>', eos_token='</s>', prefix_token='▁<PRE>', middle_token='▁<MID>', suffix_token='▁<SUF>', eot_token='▁<EOT>', fill_token='<FILL_ME>', additional_special_tokens=None, add_bos_token=True, add_eos_token=False, use_default_system_prompt=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        additional_special_tokens = additional_special_tokens or []
        for token in [prefix_token, middle_token, suffix_token, eot_token]:
            additional_special_tokens += [token] if token is not None else []
        self.use_default_system_prompt = use_default_system_prompt
        super().__init__(vocab_file=vocab_file, tokenizer_file=tokenizer_file, clean_up_tokenization_spaces=clean_up_tokenization_spaces, additional_special_tokens=additional_special_tokens, unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, add_bos_token=add_bos_token, add_eos_token=add_eos_token, prefix_token=prefix_token, middle_token=middle_token, suffix_token=suffix_token, eot_token=eot_token, fill_token=fill_token, use_default_system_prompt=use_default_system_prompt, **kwargs)
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()
        self.vocab_file = vocab_file
        self._prefix_token = prefix_token
        self._middle_token = middle_token
        self._suffix_token = suffix_token
        self._eot_token = eot_token
        self.fill_token = fill_token

    @property
    def can_save_slow_tokenizer(self) -> bool:
        if False:
            i = 10
            return i + 15
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def update_post_processor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Updates the underlying post processor with the current `bos_token` and `eos_token`.\n        '
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        if bos is None and self.add_bos_token:
            raise ValueError('add_bos_token = True but bos_token = None')
        eos = self.eos_token
        eos_token_id = self.eos_token_id
        if eos is None and self.add_eos_token:
            raise ValueError('add_eos_token = True but eos_token = None')
        single = f"{(bos + ':0 ' if self.add_bos_token else '')}$A:0{(' ' + eos + ':0' if self.add_eos_token else '')}"
        pair = f"{single}{(' ' + bos + ':1' if self.add_bos_token else '')} $B:1{(' ' + eos + ':1' if self.add_eos_token else '')}"
        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        self._tokenizer.post_processor = processors.TemplateProcessing(single=single, pair=pair, special_tokens=special_tokens)

    @property
    def prefix_token(self):
        if False:
            return 10
        return self._prefix_token

    @property
    def prefix_id(self):
        if False:
            for i in range(10):
                print('nop')
        if self._prefix_token is None:
            return None
        return self.convert_tokens_to_ids(self.prefix_token)

    @property
    def middle_token(self):
        if False:
            print('Hello World!')
        return self._middle_token

    @property
    def middle_id(self):
        if False:
            for i in range(10):
                print('nop')
        if self._middle_token is None:
            return None
        return self.convert_tokens_to_ids(self.middle_token)

    @property
    def suffix_token(self):
        if False:
            i = 10
            return i + 15
        return self._suffix_token

    @property
    def suffix_id(self):
        if False:
            i = 10
            return i + 15
        if self._suffix_token is None:
            return None
        return self.convert_tokens_to_ids(self.suffix_token)

    @property
    def eot_id(self):
        if False:
            return 10
        if self._eot_token is None:
            return None
        return self.convert_tokens_to_ids(self.eot_token)

    @property
    def eot_token(self):
        if False:
            print('Hello World!')
        return self._eot_token

    @property
    def add_eos_token(self):
        if False:
            while True:
                i = 10
        return self._add_eos_token

    @property
    def add_bos_token(self):
        if False:
            for i in range(10):
                print('nop')
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        if False:
            i = 10
            return i + 15
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._add_bos_token = value
        self.update_post_processor()

    def set_infilling_processor(self, reset, suffix_first=False, add_special_tokens=True):
        if False:
            return 10
        '\n        Updates the normalizer to make sure the prompt format for `infilling` is respected. The infilling format is the\n        following: if suffix_first\n            " <PRE> <SUF>{suf} <MID> {pre}"\n        else:\n            " <PRE> {pre} <SUF>{suf} <MID>"\n\n        If `reset` is set to `True`, the `normalizer` and `post_processor` are reset to their "normal" behaviour, which\n        is to add a prefix space for the normalizer, and add a `bos_token` to the input text for the `post_processor`.\n        '
        if reset:
            self._tokenizer.normalizer = normalizers.Sequence([normalizers.Prepend(prepend='▁'), normalizers.Replace(pattern=' ', content='▁')])
            self.update_post_processor()
            return
        self._tokenizer.normalizer = normalizers.Replace(pattern=' ', content='▁')
        pair = [self.bos_token] if self.add_bos_token and add_special_tokens else []
        special_tokens = [(self.bos_token, self.bos_token_id)] if self.add_bos_token and add_special_tokens else []
        if suffix_first:
            pair += [self.prefix_token, self.suffix_token, '$B', self.middle_token, '$A']
            special_tokens += [(self.prefix_token, self.prefix_id), (self.suffix_token, self.suffix_id), (self.middle_token, self.middle_id)]
        else:
            pair += [self.prefix_token, '$A', self.suffix_token, '$B', self.middle_token]
            special_tokens += [(self.prefix_token, self.prefix_id), (self.suffix_token, self.suffix_id), (self.middle_token, self.middle_id)]
        if self.add_eos_token and add_special_tokens:
            pair += [self.eos_token]
            special_tokens += [(self.eos_token, self.eos_token_id)]
        self._tokenizer.post_processor = processors.TemplateProcessing(single='$A', pair=pair, special_tokens=special_tokens)

    def encode_plus(self, text, text_pair=None, suffix_first=False, add_special_tokens=True, **kwargs):
        if False:
            i = 10
            return i + 15
        text_pair = kwargs.pop('suffix', text_pair)
        if self.fill_token is not None and self.fill_token in text and (text_pair is None):
            (text, text_pair) = text.split(self.fill_token)
        if text_pair is None or len(text_pair) < 1:
            return super().encode_plus(text, text_pair, add_special_tokens=add_special_tokens, **kwargs)
        if None in (self.prefix_id, self.middle_id, self.suffix_id):
            raise ValueError(f'Then input includes a `prefix` and a `suffix` used for the infilling task, the `prefix_id, middle_id, suffix_id` must all be initialized. Current values : {(self.prefix_id, self.middle_id, self.suffix_id)}')
        self.set_infilling_processor(False, suffix_first=suffix_first, add_special_tokens=add_special_tokens)
        tokens = super().encode_plus(' ' + text, text_pair=text_pair, add_special_tokens=True, **kwargs)
        self.set_infilling_processor(True)
        return tokens

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        if False:
            print('Hello World!')
        if not self.can_save_slow_tokenizer:
            raise ValueError('Your fast tokenizer does not have the necessary information to save the vocabulary for a slow tokenizer.')
        if not os.path.isdir(save_directory):
            logger.error(f'Vocabulary path ({save_directory}) should be a directory')
            return
        out_vocab_file = os.path.join(save_directory, (filename_prefix + '-' if filename_prefix else '') + VOCAB_FILES_NAMES['vocab_file'])
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        return (out_vocab_file,)

    @property
    def default_chat_template(self):
        if False:
            i = 10
            return i + 15
        "\n        LLaMA uses [INST] and [/INST] to indicate user messages, and <<SYS>> and <</SYS>> to indicate system messages.\n        Assistant messages do not have special tokens, because LLaMA chat models are generally trained with strict\n        user/assistant/user/assistant message ordering, and so assistant messages can be identified from the ordering\n        rather than needing special tokens. The system message is partly 'embedded' in the first user message, which\n        results in an unusual token ordering when it is present. This template should definitely be changed if you wish\n        to fine-tune a model with more flexible role ordering!\n\n        The output should look something like:\n\n        <bos>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST] Answer <eos><bos>[INST] Prompt [/INST] Answer <eos>\n        <bos>[INST] Prompt [/INST]\n\n        The reference for this chat template is [this code\n        snippet](https://github.com/facebookresearch/llama/blob/556949fdfb72da27c2f4a40b7f0e4cf0b8153a28/llama/generation.py#L320-L362)\n        in the original repository.\n        "
        logger.warning_once(f'\nNo chat template is defined for this tokenizer - using the default template for the {self.__class__.__name__} class. If the default is not appropriate for your model, please set `tokenizer.chat_template` to an appropriate template. See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n')
        template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        template = template.replace('USE_DEFAULT_PROMPT', 'true' if self.use_default_system_prompt else 'false')
        default_message = DEFAULT_SYSTEM_PROMPT.replace('\n', '\\n').replace("'", "\\'")
        template = template.replace('DEFAULT_SYSTEM_MESSAGE', default_message)
        return template

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if False:
            i = 10
            return i + 15
        '\n        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and\n        adding special tokens. The special tokens depend on calling set_lang.\n\n        An NLLB sequence has the following format, where `X` represents the sequence:\n\n        - `input_ids` (for encoder) `X [eos, src_lang_code]`\n        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`\n\n        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a\n        separator.\n\n        Args:\n            token_ids_0 (`List[int]`):\n                List of IDs to which the special tokens will be added.\n            token_ids_1 (`List[int]`, *optional*):\n                Optional second list of IDs for sequence pairs.\n\n        Returns:\n            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.\n        '
        if token_ids_1 is None:
            return self.bos_token_id + token_ids_0 + self.eos_token_id
        return self.bos_token_id + token_ids_0 + token_ids_1 + self.eos_token_id
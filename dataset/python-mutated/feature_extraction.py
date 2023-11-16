from typing import Dict, Any, Union, Tuple, Optional, List
import re
import os
import json
import logging
from pathlib import Path
import numpy as np
from haystack.errors import ModelingError
from haystack.modeling.data_handler.samples import SampleBasket
from haystack.lazy_imports import LazyImport
logger = logging.getLogger(__name__)
SPECIAL_TOKENIZER_CHARS = '^(##|Ġ|▁)'
with LazyImport(message="Run 'pip install farm-haystack[inference]'") as transformers_import:
    import transformers
    from transformers import PreTrainedTokenizer, RobertaTokenizer, AutoConfig, AutoFeatureExtractor, AutoTokenizer
    from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING_NAMES
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
    FEATURE_EXTRACTORS = {**{key: AutoTokenizer for key in TOKENIZER_MAPPING_NAMES.keys()}, **{key: AutoFeatureExtractor for key in FEATURE_EXTRACTOR_MAPPING_NAMES.keys()}}
    DEFAULT_EXTRACTION_PARAMS = {AutoTokenizer: {'max_length': 256, 'add_special_tokens': True, 'truncation': True, 'truncation_strategy': 'longest_first', 'padding': 'max_length', 'return_token_type_ids': True}, AutoFeatureExtractor: {'return_tensors': 'pt'}}

class FeatureExtractor:

    def __init__(self, pretrained_model_name_or_path: Union[str, Path], revision: Optional[str]=None, use_fast: bool=True, use_auth_token: Optional[Union[str, bool]]=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Enables loading of different feature extractors, including tokenizers, with a uniform interface.\n\n        Use `FeatureExtractor.extract_features()` to convert your input queries, documents, images, and tables\n        into vectors that you can pass to the language model.\n\n        :param pretrained_model_name_or_path:  The path of the saved pretrained model or its name (for example, `bert-base-uncased`)\n        :param revision: The version of the model to use from the Hugging Face model hub. It can be tag name, branch name, or commit hash.\n        :param use_fast: Indicate if Haystack should try to load the fast version of the tokenizer (True) or use the Python one (False). Defaults to True.\n        :param use_auth_token: The API token used to download private models from Hugging Face.\n                            If this parameter is set to `True`, then the token generated when running\n                            `transformers-cli login` (stored in ~/.huggingface) is used.\n                            For more information, see\n                            [Hugging Face documentation](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained)\n        :param kwargs: Other kwargs you want to pass on to `PretrainedTokenizer.from_pretrained()`\n        '
        transformers_import.check()
        model_name_or_path = str(pretrained_model_name_or_path)
        model_type = None
        config_file = Path(pretrained_model_name_or_path) / 'tokenizer_config.json'
        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
            feature_extractor_classname = config['tokenizer_class']
            logger.debug('⛏️ Selected feature extractor: %s (from %s)', feature_extractor_classname, config_file)
            try:
                feature_extractor_class = getattr(transformers, feature_extractor_classname + 'Fast')
                logger.debug('Fast version of this tokenizer exists. Loaded class: %s', feature_extractor_class.__class__.__name__)
            except AttributeError:
                logger.debug('Fast version could not be loaded. Falling back to base version.')
                feature_extractor_class = getattr(transformers, feature_extractor_classname)
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path, use_auth_token=use_auth_token, revision=revision)
            model_type = config.model_type
            try:
                feature_extractor_class = FEATURE_EXTRACTORS[model_type]
            except KeyError as e:
                raise ModelingError(f"'{pretrained_model_name_or_path}' has no known feature extractor. Haystack can assign tokenizers to the following model types: \n- {f'{chr(10)}- '.join(FEATURE_EXTRACTORS.keys())}") from e
            logger.debug("⛏️ Selected feature extractor: %s (for model type '%s')", feature_extractor_class.__name__, model_type)
        self.default_params = DEFAULT_EXTRACTION_PARAMS.get(feature_extractor_class, {})
        self.feature_extractor = feature_extractor_class.from_pretrained(pretrained_model_name_or_path=model_name_or_path, revision=revision, use_fast=use_fast, use_auth_token=use_auth_token, **kwargs)

    def __call__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        params = {**self.default_params, **(kwargs or {})}
        return self.feature_extractor(**params)

def tokenize_batch_question_answering(pre_baskets: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, indices: List[Any]) -> List[SampleBasket]:
    if False:
        return 10
    "\n    Tokenizes text data for question answering tasks. Tokenization means splitting words into subwords, depending on the\n    tokenizer's vocabulary.\n\n    - We first tokenize all documents in batch mode. (When using FastTokenizers Rust multithreading can be enabled by TODO add how to enable rust mt)\n    - Then we tokenize each question individually\n    - We construct dicts with question and corresponding document text + tokens + offsets + ids\n\n    :param pre_baskets: input dicts with QA info #TODO change to input objects\n    :param tokenizer: tokenizer to be used\n    :param indices: indices used during multiprocessing so that IDs assigned to our baskets are unique\n    :return: baskets, list containing question and corresponding document information\n    "
    if not len(indices) == len(pre_baskets):
        raise ValueError('indices and pre_baskets must have the same length')
    if not tokenizer.is_fast:
        raise ModelingError("Processing QA data is only supported with fast tokenizers for now.Please load Tokenizers with 'use_fast=True' option.")
    baskets = []
    texts = [d['context'] for d in pre_baskets]
    tokenized_docs_batch = tokenizer(text=texts, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=False, verbose=False)
    tokenids_batch = tokenized_docs_batch['input_ids']
    offsets_batch = []
    for o in tokenized_docs_batch['offset_mapping']:
        offsets_batch.append(np.asarray([x[0] for x in o], dtype=np.int32))
    start_of_words_batch = []
    for e in tokenized_docs_batch.encodings:
        start_of_words_batch.append(_get_start_of_word_QA(e.word_ids))
    for (i_doc, d) in enumerate(pre_baskets):
        document_text = d['context']
        for (i_q, q) in enumerate(d['qas']):
            question_text = q['question']
            tokenized_q = tokenizer(question_text, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=False)
            question_tokenids = tokenized_q['input_ids']
            question_offsets = [x[0] for x in tokenized_q['offset_mapping']]
            question_sow = _get_start_of_word_QA(tokenized_q.encodings[0].word_ids)
            external_id = q['id']
            internal_id = f'{indices[i_doc]}-{i_q}'
            raw = {'document_text': document_text, 'document_tokens': tokenids_batch[i_doc], 'document_offsets': offsets_batch[i_doc], 'document_start_of_word': start_of_words_batch[i_doc], 'question_text': question_text, 'question_tokens': question_tokenids, 'question_offsets': question_offsets, 'question_start_of_word': question_sow, 'answers': q['answers']}
            raw['document_tokens_strings'] = tokenized_docs_batch.encodings[i_doc].tokens
            raw['question_tokens_strings'] = tokenized_q.encodings[0].tokens
            baskets.append(SampleBasket(raw=raw, id_internal=internal_id, id_external=external_id, samples=None))
    return baskets

def _get_start_of_word_QA(word_ids):
    if False:
        print('Hello World!')
    return [1] + list(np.ediff1d(np.asarray(word_ids, dtype=np.int32)))

def truncate_sequences(seq_a: list, seq_b: Optional[list], tokenizer: AutoTokenizer, max_seq_len: int, truncation_strategy: str='longest_first', with_special_tokens: bool=True, stride: int=0) -> Tuple[List[Any], Optional[List[Any]], List[Any]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Reduces a single sequence or a pair of sequences to a maximum sequence length.\n    The sequences can contain tokens or any other elements (offsets, masks ...).\n    If `with_special_tokens` is enabled, it\'ll remove some additional tokens to have exactly\n    enough space for later adding special tokens (CLS, SEP etc.)\n\n    Supported truncation strategies:\n\n    - longest_first: (default) Iteratively reduce the inputs sequence until the input is under\n        max_length starting from the longest one at each token (when there is a pair of input sequences).\n        Overflowing tokens only contains overflow from the first sequence.\n    - only_first: Only truncate the first sequence. raise an error if the first sequence is\n        shorter or equal to than num_tokens_to_remove.\n    - only_second: Only truncate the second sequence\n    - do_not_truncate: Does not truncate (raise an error if the input sequence is longer than max_length)\n\n    :param seq_a: First sequence of tokens/offsets/...\n    :param seq_b: Optional second sequence of tokens/offsets/...\n    :param tokenizer: Tokenizer (e.g. from get_tokenizer))\n    :param max_seq_len:\n    :param truncation_strategy: how the sequence(s) should be truncated down.\n        Default: "longest_first" (see above for other options).\n    :param with_special_tokens: If true, it\'ll remove some additional tokens to have exactly enough space\n        for later adding special tokens (CLS, SEP etc.)\n    :param stride: optional stride of the window during truncation\n    :return: truncated seq_a, truncated seq_b, overflowing tokens\n    '
    pair = seq_b is not None
    len_a = len(seq_a)
    len_b = len(seq_b) if seq_b is not None else 0
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=pair) if with_special_tokens else 0
    total_len = len_a + len_b + num_special_tokens
    overflowing_tokens = []
    if max_seq_len and total_len > max_seq_len:
        (seq_a, seq_b, overflowing_tokens) = tokenizer.truncate_sequences(seq_a, pair_ids=seq_b, num_tokens_to_remove=total_len - max_seq_len, truncation_strategy=truncation_strategy, stride=stride)
    return (seq_a, seq_b, overflowing_tokens)

def tokenize_with_metadata(text: str, tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    if False:
        return 10
    '\n    Performing tokenization while storing some important metadata for each token:\n\n    * offsets: (int) Character index where the token begins in the original text\n    * start_of_word: (bool) If the token is the start of a word. Particularly helpful for NER and QA tasks.\n\n    We do this by first doing whitespace tokenization and then applying the model specific tokenizer to each "word".\n\n    .. note::  We don\'t assume to preserve exact whitespaces in the tokens!\n               This means: tabs, new lines, multiple whitespace etc will all resolve to a single " ".\n               This doesn\'t make a difference for BERT + XLNet but it does for RoBERTa.\n               For RoBERTa it has the positive effect of a shorter sequence length, but some information about whitespace\n               type is lost which might be helpful for certain NLP tasks ( e.g tab for tables).\n\n    :param text: Text to tokenize\n    :param tokenizer: Tokenizer (e.g. from get_tokenizer))\n    :return: Dictionary with "tokens", "offsets" and "start_of_word"\n    '
    text = re.sub('\\s', ' ', text)
    words: Union[List[str], np.ndarray] = []
    word_offsets: Union[List[int], np.ndarray] = []
    start_of_word: List[Union[int, bool]] = []
    if tokenizer.is_fast:
        tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
        tokens = tokenized['input_ids']
        offsets = np.array([x[0] for x in tokenized['offset_mapping']])
        words = np.array(tokenized.encodings[0].words)
        words[0] = -1
        words[-1] = words[-2]
        words += 1
        start_of_word = [0] + list(np.ediff1d(words))
        return {'tokens': tokens, 'offsets': offsets, 'start_of_word': start_of_word}
    words = text.split(' ')
    cumulated = 0
    for word in words:
        word_offsets.append(cumulated)
        cumulated += len(word) + 1
    (tokens, offsets, start_of_word) = _words_to_tokens(words, word_offsets, tokenizer)
    return {'tokens': tokens, 'offsets': offsets, 'start_of_word': start_of_word}

def _words_to_tokens(words: List[str], word_offsets: List[int], tokenizer: PreTrainedTokenizer) -> Tuple[List[str], List[int], List[bool]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Tokenize "words" into subword tokens while keeping track of offsets and if a token is the start of a word.\n    :param words: list of words.\n    :param word_offsets: Character indices where each word begins in the original text\n    :param tokenizer: Tokenizer (e.g. from get_tokenizer))\n    :return: Tuple of (tokens, offsets, start_of_word)\n    '
    tokens: List[str] = []
    token_offsets: List[int] = []
    start_of_word: List[bool] = []
    index = 0
    for (index, (word, word_offset)) in enumerate(zip(words, word_offsets)):
        if index % 500000 == 0:
            logger.info(index)
        if len(word) == 0:
            continue
        if len(tokens) == 0:
            tokens_word = tokenizer.tokenize(word)
        elif type(tokenizer) == RobertaTokenizer:
            tokens_word = tokenizer.tokenize(word, add_prefix_space=True)
        else:
            tokens_word = tokenizer.tokenize(word)
        if len(tokens_word) == 0:
            continue
        tokens += tokens_word
        first_token = True
        for token in tokens_word:
            token_offsets.append(word_offset)
            original_token = re.sub(SPECIAL_TOKENIZER_CHARS, '', token)
            if original_token == tokenizer.special_tokens_map['unk_token']:
                word_offset += 1
            else:
                word_offset += len(original_token)
            if first_token:
                start_of_word.append(True)
                first_token = False
            else:
                start_of_word.append(False)
    return (tokens, token_offsets, start_of_word)
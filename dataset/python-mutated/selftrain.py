"""
Common methods for the various self-training data collection scripts
"""
import logging
import os
import random
import re
import stanza
from stanza.models.common import utils
from stanza.models.common.bert_embedding import TextTooLongError
from stanza.utils.get_tqdm import get_tqdm
logger = logging.getLogger('stanza')
tqdm = get_tqdm()

def common_args(parser):
    if False:
        for i in range(10):
            print('nop')
    parser.add_argument('--output_file', default='data/constituency/vi_silver.mrg', help='Where to write the silver trees')
    parser.add_argument('--lang', default='vi', help='Which language tools to use for tokenization and POS')
    parser.add_argument('--num_sentences', type=int, default=-1, help='How many sentences to get per file (max)')
    parser.add_argument('--models', default='saved_models/constituency/vi_vlsp21_inorder.pt', help='What models to use for parsing.  comma-separated')
    parser.add_argument('--package', default='default', help='Which package to load pretrain & charlm from for the parsers')
    parser.add_argument('--output_ptb', default=False, action='store_true', help='Output trees in PTB brackets (default is a bracket language format)')

def add_length_args(parser):
    if False:
        print('Hello World!')
    parser.add_argument('--min_len', default=5, type=int, help='Minimum length sentence to keep.  None = unlimited')
    parser.add_argument('--no_min_len', dest='min_len', action='store_const', const=None, help='No minimum length')
    parser.add_argument('--max_len', default=100, type=int, help='Maximum length sentence to keep.  None = unlimited')
    parser.add_argument('--no_max_len', dest='max_len', action='store_const', const=None, help='No maximum length')

def build_ssplit_pipe(ssplit, lang):
    if False:
        print('Hello World!')
    if ssplit:
        return stanza.Pipeline(lang, processors='tokenize')
    else:
        return stanza.Pipeline(lang, processors='tokenize', tokenize_no_ssplit=True)

def build_tag_pipe(ssplit, lang, foundation_cache=None):
    if False:
        while True:
            i = 10
    if ssplit:
        return stanza.Pipeline(lang, processors='tokenize,pos', foundation_cache=foundation_cache)
    else:
        return stanza.Pipeline(lang, processors='tokenize,pos', tokenize_no_ssplit=True, foundation_cache=foundation_cache)

def build_parser_pipes(lang, models, package='default', foundation_cache=None):
    if False:
        i = 10
        return i + 15
    '\n    Build separate pipelines for each parser model we want to use\n\n    It is highly recommended to pass in a FoundationCache to reuse bottom layers\n    '
    parser_pipes = []
    for model_name in models.split(','):
        if os.path.exists(model_name):
            pipe = stanza.Pipeline(lang, processors='constituency', package=package, constituency_model_path=model_name, constituency_pretagged=True, foundation_cache=foundation_cache)
        else:
            pipe = stanza.Pipeline(lang, processors={'constituency': model_name}, constituency_pretagged=True, package=None, foundation_cache=foundation_cache)
        parser_pipes.append(pipe)
    return parser_pipes

def split_docs(docs, ssplit_pipe, max_len=140, max_word_len=50, chunk_size=2000):
    if False:
        return 10
    '\n    Using the ssplit pipeline, break up the documents into sentences\n\n    Filters out sentences which are too long or have words too long.\n\n    This step is necessary because some web text has unstructured\n    sentences which overwhelm the tagger, or even text with no\n    whitespace which breaks the charlm in the tokenizer or tagger\n    '
    raw_sentences = 0
    filtered_sentences = 0
    new_docs = []
    logger.info('Splitting raw docs into sentences: %d', len(docs))
    for chunk_start in tqdm(range(0, len(docs), chunk_size)):
        chunk = docs[chunk_start:chunk_start + chunk_size]
        chunk = [stanza.Document([], text=t) for t in chunk]
        chunk = ssplit_pipe(chunk)
        sentences = [s for d in chunk for s in d.sentences]
        raw_sentences += len(sentences)
        sentences = [s for s in sentences if len(s.words) < max_len]
        sentences = [s for s in sentences if max((len(w.text) for w in s.words)) < max_word_len]
        filtered_sentences += len(sentences)
        new_docs.extend([s.text for s in sentences])
    logger.info('Split sentences: %d', raw_sentences)
    logger.info('Sentences filtered for length: %d', filtered_sentences)
    return new_docs
ZH_RE = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
JA_RE = re.compile(u'[一-龠ぁ-ゔァ-ヴー々〆〤ヶ]', re.UNICODE)
DEV_RE = re.compile(u'[ऀ-ॿ]', re.UNICODE)

def tokenize_docs(docs, pipe, min_len, max_len):
    if False:
        for i in range(10):
            print('nop')
    '\n    Turn the text in docs into a list of whitespace separated sentences\n\n    docs: a list of strings\n    pipe: a Stanza pipeline for tokenizing\n    min_len, max_len: can be None to not filter by this attribute\n    '
    results = []
    docs = [stanza.Document([], text=t) for t in docs]
    if len(docs) == 0:
        return results
    pipe(docs)
    is_zh = pipe.lang and pipe.lang.startswith('zh')
    is_ja = pipe.lang and pipe.lang.startswith('ja')
    is_vi = pipe.lang and pipe.lang.startswith('vi')
    for doc in docs:
        for sentence in doc.sentences:
            if min_len and len(sentence.words) < min_len:
                continue
            if max_len and len(sentence.words) > max_len:
                continue
            text = sentence.text
            if text.find('|') >= 0 or text.find('_') >= 0 or text.find('<') >= 0 or (text.find('>') >= 0) or (text.find('[') >= 0) or (text.find(']') >= 0) or (text.find('—') >= 0):
                continue
            if any((any((w.text.find(c) >= 0 and len(w.text) > 1 for w in sentence.words)) for c in '"()')):
                continue
            text = [w.text.replace(' ', '_') for w in sentence.words]
            text = ' '.join(text)
            if any((len(w.text) >= 50 for w in sentence.words)):
                continue
            if not is_zh and len(ZH_RE.findall(text)) > 250:
                continue
            if not is_ja and len(JA_RE.findall(text)) > 150:
                continue
            if is_vi and len(DEV_RE.findall(text)) > 100:
                continue
            results.append(text)
    return results

def find_matching_trees(docs, num_sentences, accepted_trees, tag_pipe, parser_pipes, shuffle=True, chunk_size=10, max_len=140, min_len=10, output_ptb=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find trees where all the parsers in parser_pipes agree\n\n    docs should be a list of strings.\n      one sentence per string or a whole block of text as long as the tag_pipe can break it into sentences\n\n    num_sentences > 0 gives an upper limit on how many sentences to extract.\n      If < 0, all possible sentences are extracted\n\n    accepted_trees is a running tally of all the trees already built,\n      so that we don't reuse the same sentence if we see it again\n    "
    if num_sentences < 0:
        tqdm_total = len(docs)
    else:
        tqdm_total = num_sentences
    if output_ptb:
        output_format = '{}'
    else:
        output_format = '{:L}'
    with tqdm(total=tqdm_total, leave=False) as pbar:
        if shuffle:
            random.shuffle(docs)
        new_trees = set()
        for chunk_start in range(0, len(docs), chunk_size):
            chunk = docs[chunk_start:chunk_start + chunk_size]
            chunk = [stanza.Document([], text=t) for t in chunk]
            if num_sentences < 0:
                pbar.update(len(chunk))
            tag_pipe(chunk)
            chunk = [d for d in chunk if len(d.sentences) > 0]
            if max_len is not None:
                chunk = [d for d in chunk if max((len(s.words) for s in d.sentences)) < max_len]
            if len(chunk) == 0:
                continue
            parses = []
            try:
                for pipe in parser_pipes:
                    pipe(chunk)
                    trees = [output_format.format(sent.constituency) for doc in chunk for sent in doc.sentences if len(sent.words) >= min_len]
                    parses.append(trees)
            except TextTooLongError as e:
                continue
            for tree in zip(*parses):
                if len(set(tree)) != 1:
                    continue
                tree = tree[0]
                if tree in accepted_trees:
                    continue
                if tree not in new_trees:
                    new_trees.add(tree)
                    if num_sentences >= 0:
                        pbar.update(1)
                if num_sentences >= 0 and len(new_trees) >= num_sentences:
                    return new_trees
    return new_trees
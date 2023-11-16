"""Construct a corpus from a Wikipedia (or other MediaWiki-based) database dump.

Uses multiprocessing internally to parallelize the work and process the dump more quickly.

Notes
-----

See :mod:`gensim.scripts.make_wiki` for a canned (example) command-line script based on this module.
"""
import bz2
import logging
import multiprocessing
import re
import signal
from pickle import PicklingError
from xml.etree.ElementTree import iterparse
from gensim import utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
logger = logging.getLogger(__name__)
ARTICLE_MIN_WORDS = 50
'Ignore shorter articles (after full preprocessing).'
TOKEN_MIN_LEN = 2
TOKEN_MAX_LEN = 15
RE_P0 = re.compile('<!--.*?-->', re.DOTALL | re.UNICODE)
'Comments.'
RE_P1 = re.compile('<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)
'Footnotes.'
RE_P2 = re.compile('(\\n\\[\\[[a-z][a-z][\\w-]*:[^:\\]]+\\]\\])+$', re.UNICODE)
'Links to languages.'
RE_P3 = re.compile('{{([^}{]*)}}', re.DOTALL | re.UNICODE)
'Template.'
RE_P4 = re.compile('{{([^}]*)}}', re.DOTALL | re.UNICODE)
'Template.'
RE_P5 = re.compile('\\[(\\w+):\\/\\/(.*?)(( (.*?))|())\\]', re.UNICODE)
'Remove URL, keep description.'
RE_P6 = re.compile('\\[([^][]*)\\|([^][]*)\\]', re.DOTALL | re.UNICODE)
'Simplify links, keep description.'
RE_P7 = re.compile('\\n\\[\\[[iI]mage(.*?)(\\|.*?)*\\|(.*?)\\]\\]', re.UNICODE)
'Keep description of images.'
RE_P8 = re.compile('\\n\\[\\[[fF]ile(.*?)(\\|.*?)*\\|(.*?)\\]\\]', re.UNICODE)
'Keep description of files.'
RE_P9 = re.compile('<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE)
'External links.'
RE_P10 = re.compile('<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE)
'Math content.'
RE_P11 = re.compile('<(.*?)>', re.DOTALL | re.UNICODE)
'All other tags.'
RE_P12 = re.compile('(({\\|)|(\\|-(?!\\d))|(\\|}))(.*?)(?=\\n)', re.UNICODE)
'Table formatting.'
RE_P13 = re.compile('(?<=(\\n[ ])|(\\n\\n)|([ ]{2})|(.\\n)|(.\\t))(\\||\\!)([^[\\]\\n]*?\\|)*', re.UNICODE)
'Table cell formatting.'
RE_P14 = re.compile('\\[\\[Category:[^][]*\\]\\]', re.UNICODE)
'Categories.'
RE_P15 = re.compile('\\[\\[([fF]ile:|[iI]mage)[^]]*(\\]\\])', re.UNICODE)
'Remove File and Image templates.'
RE_P16 = re.compile('\\[{2}(.*?)\\]{2}', re.UNICODE)
'Capture interlinks text and article linked'
RE_P17 = re.compile('(\\n.{0,4}((bgcolor)|(\\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=)|(scope=))(.*))|(^.{0,2}((bgcolor)|(\\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=))(.*))', re.UNICODE)
'Table markup'
IGNORED_NAMESPACES = ['Wikipedia', 'Category', 'File', 'Portal', 'Template', 'MediaWiki', 'User', 'Help', 'Book', 'Draft', 'WikiProject', 'Special', 'Talk']
'MediaWiki namespaces that ought to be ignored.'

def filter_example(elem, text, *args, **kwargs):
    if False:
        while True:
            i = 10
    "Example function for filtering arbitrary documents from wikipedia dump.\n\n\n    The custom filter function is called _before_ tokenisation and should work on\n    the raw text and/or XML element information.\n\n    The filter function gets the entire context of the XML element passed into it,\n    but you can of course choose not the use some or all parts of the context. Please\n    refer to :func:`gensim.corpora.wikicorpus.extract_pages` for the exact details\n    of the page context.\n\n    Parameters\n    ----------\n    elem : etree.Element\n        XML etree element\n    text : str\n        The text of the XML node\n    namespace : str\n        XML namespace of the XML element\n    title : str\n       Page title\n    page_tag : str\n        XPath expression for page.\n    text_path : str\n        XPath expression for text.\n    title_path : str\n        XPath expression for title.\n    ns_path : str\n        XPath expression for namespace.\n    pageid_path : str\n        XPath expression for page id.\n\n    Example\n    -------\n    .. sourcecode:: pycon\n\n        >>> import gensim.corpora\n        >>> filter_func = gensim.corpora.wikicorpus.filter_example\n        >>> dewiki = gensim.corpora.WikiCorpus(\n        ...     './dewiki-20180520-pages-articles-multistream.xml.bz2',\n        ...     filter_articles=filter_func)\n\n    "
    _regex_de_excellent = re.compile('.*\\{\\{(Exzellent.*?)\\}\\}[\\s]*', flags=re.DOTALL)
    _regex_de_featured = re.compile('.*\\{\\{(Lesenswert.*?)\\}\\}[\\s]*', flags=re.DOTALL)
    if text is None:
        return False
    if _regex_de_excellent.match(text) or _regex_de_featured.match(text):
        return True
    else:
        return False

def find_interlinks(raw):
    if False:
        print('Hello World!')
    'Find all interlinks to other articles in the dump.\n\n    Parameters\n    ----------\n    raw : str\n        Unicode or utf-8 encoded string.\n\n    Returns\n    -------\n    list\n        List of tuples in format [(linked article, the actual text found), ...].\n\n    '
    filtered = filter_wiki(raw, promote_remaining=False, simplify_links=False)
    interlinks_raw = re.findall(RE_P16, filtered)
    interlinks = []
    for parts in [i.split('|') for i in interlinks_raw]:
        actual_title = parts[0]
        try:
            interlink_text = parts[1]
        except IndexError:
            interlink_text = actual_title
        interlink_tuple = (actual_title, interlink_text)
        interlinks.append(interlink_tuple)
    legit_interlinks = [(i, j) for (i, j) in interlinks if '[' not in i and ']' not in i]
    return legit_interlinks

def filter_wiki(raw, promote_remaining=True, simplify_links=True):
    if False:
        print('Hello World!')
    'Filter out wiki markup from `raw`, leaving only text.\n\n    Parameters\n    ----------\n    raw : str\n        Unicode or utf-8 encoded string.\n    promote_remaining : bool\n        Whether uncaught markup should be promoted to plain text.\n    simplify_links : bool\n        Whether links should be simplified keeping only their description text.\n\n    Returns\n    -------\n    str\n        `raw` without markup.\n\n    '
    text = utils.to_unicode(raw, 'utf8', errors='ignore')
    text = utils.decode_htmlentities(text)
    return remove_markup(text, promote_remaining, simplify_links)

def remove_markup(text, promote_remaining=True, simplify_links=True):
    if False:
        return 10
    'Filter out wiki markup from `text`, leaving only text.\n\n    Parameters\n    ----------\n    text : str\n        String containing markup.\n    promote_remaining : bool\n        Whether uncaught markup should be promoted to plain text.\n    simplify_links : bool\n        Whether links should be simplified keeping only their description text.\n\n    Returns\n    -------\n    str\n        `text` without markup.\n\n    '
    text = re.sub(RE_P2, '', text)
    text = remove_template(text)
    text = remove_file(text)
    iters = 0
    while True:
        (old, iters) = (text, iters + 1)
        text = re.sub(RE_P0, '', text)
        text = re.sub(RE_P1, '', text)
        text = re.sub(RE_P9, '', text)
        text = re.sub(RE_P10, '', text)
        text = re.sub(RE_P11, '', text)
        text = re.sub(RE_P14, '', text)
        text = re.sub(RE_P5, '\\3', text)
        if simplify_links:
            text = re.sub(RE_P6, '\\2', text)
        text = text.replace('!!', '\n|')
        text = text.replace('|-||', '\n|')
        text = re.sub(RE_P12, '\n', text)
        text = text.replace('|||', '|\n|')
        text = text.replace('||', '\n|')
        text = re.sub(RE_P13, '\n', text)
        text = re.sub(RE_P17, '\n', text)
        text = text.replace('[]', '')
        if old == text or iters > 2:
            break
    if promote_remaining:
        text = text.replace('[', '').replace(']', '')
    return text

def remove_template(s):
    if False:
        for i in range(10):
            print('nop')
    'Remove template wikimedia markup.\n\n    Parameters\n    ----------\n    s : str\n        String containing markup template.\n\n    Returns\n    -------\n    str\n        Сopy of `s` with all the `wikimedia markup template <http://meta.wikimedia.org/wiki/Help:Template>`_ removed.\n\n    Notes\n    -----\n    Since template can be nested, it is difficult remove them using regular expressions.\n\n    '
    (n_open, n_close) = (0, 0)
    (starts, ends) = ([], [-1])
    in_template = False
    prev_c = None
    for (i, c) in enumerate(s):
        if not in_template:
            if c == '{' and c == prev_c:
                starts.append(i - 1)
                in_template = True
                n_open = 1
        if in_template:
            if c == '{':
                n_open += 1
            elif c == '}':
                n_close += 1
            if n_open == n_close:
                ends.append(i)
                in_template = False
                (n_open, n_close) = (0, 0)
        prev_c = c
    starts.append(None)
    return ''.join((s[end + 1:start] for (end, start) in zip(ends, starts)))

def remove_file(s):
    if False:
        print('Hello World!')
    "Remove the 'File:' and 'Image:' markup, keeping the file caption.\n\n    Parameters\n    ----------\n    s : str\n        String containing 'File:' and 'Image:' markup.\n\n    Returns\n    -------\n    str\n        Сopy of `s` with all the 'File:' and 'Image:' markup replaced by their `corresponding captions\n        <http://www.mediawiki.org/wiki/Help:Images>`_.\n\n    "
    for match in re.finditer(RE_P15, s):
        m = match.group(0)
        caption = m[:-2].split('|')[-1]
        s = s.replace(m, caption, 1)
    return s

def tokenize(content, token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True):
    if False:
        i = 10
        return i + 15
    'Tokenize a piece of text from Wikipedia.\n\n    Set `token_min_len`, `token_max_len` as character length (not bytes!) thresholds for individual tokens.\n\n    Parameters\n    ----------\n    content : str\n        String without markup (see :func:`~gensim.corpora.wikicorpus.filter_wiki`).\n    token_min_len : int\n        Minimal token length.\n    token_max_len : int\n        Maximal token length.\n    lower : bool\n         Convert `content` to lower case?\n\n    Returns\n    -------\n    list of str\n        List of tokens from `content`.\n\n    '
    return [utils.to_unicode(token) for token in utils.tokenize(content, lower=lower, errors='ignore') if token_min_len <= len(token) <= token_max_len and (not token.startswith('_'))]

def get_namespace(tag):
    if False:
        for i in range(10):
            print('nop')
    'Get the namespace of tag.\n\n    Parameters\n    ----------\n    tag : str\n        Namespace or tag.\n\n    Returns\n    -------\n    str\n        Matched namespace or tag.\n\n    '
    m = re.match('^{(.*?)}', tag)
    namespace = m.group(1) if m else ''
    if not namespace.startswith('http://www.mediawiki.org/xml/export-'):
        raise ValueError('%s not recognized as MediaWiki dump namespace' % namespace)
    return namespace
_get_namespace = get_namespace

def extract_pages(f, filter_namespaces=False, filter_articles=None):
    if False:
        i = 10
        return i + 15
    'Extract pages from a MediaWiki database dump.\n\n    Parameters\n    ----------\n    f : file\n        File-like object.\n    filter_namespaces : list of str or bool\n         Namespaces that will be extracted.\n\n    Yields\n    ------\n    tuple of (str or None, str, str)\n        Title, text and page id.\n\n    '
    elems = (elem for (_, elem) in iterparse(f, events=('end',)))
    elem = next(elems)
    namespace = get_namespace(elem.tag)
    ns_mapping = {'ns': namespace}
    page_tag = '{%(ns)s}page' % ns_mapping
    text_path = './{%(ns)s}revision/{%(ns)s}text' % ns_mapping
    title_path = './{%(ns)s}title' % ns_mapping
    ns_path = './{%(ns)s}ns' % ns_mapping
    pageid_path = './{%(ns)s}id' % ns_mapping
    for elem in elems:
        if elem.tag == page_tag:
            title = elem.find(title_path).text
            text = elem.find(text_path).text
            if filter_namespaces:
                ns = elem.find(ns_path).text
                if ns not in filter_namespaces:
                    text = None
            if filter_articles is not None:
                if not filter_articles(elem, namespace=namespace, title=title, text=text, page_tag=page_tag, text_path=text_path, title_path=title_path, ns_path=ns_path, pageid_path=pageid_path):
                    text = None
            pageid = elem.find(pageid_path).text
            yield (title, text or '', pageid)
            elem.clear()
_extract_pages = extract_pages

def process_article(args, tokenizer_func=tokenize, token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True):
    if False:
        return 10
    'Parse a Wikipedia article, extract all tokens.\n\n    Notes\n    -----\n    Set `tokenizer_func` (defaults is :func:`~gensim.corpora.wikicorpus.tokenize`) parameter for languages\n    like Japanese or Thai to perform better tokenization.\n    The `tokenizer_func` needs to take 4 parameters: (text: str, token_min_len: int, token_max_len: int, lower: bool).\n\n    Parameters\n    ----------\n    args : (str, str, int)\n        Article text, article title, page identificator.\n    tokenizer_func : function\n        Function for tokenization (defaults is :func:`~gensim.corpora.wikicorpus.tokenize`).\n        Needs to have interface:\n        tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list of str.\n    token_min_len : int\n        Minimal token length.\n    token_max_len : int\n        Maximal token length.\n    lower : bool\n         Convert article text to lower case?\n\n    Returns\n    -------\n    (list of str, str, int)\n        List of tokens from article, title and page id.\n\n    '
    (text, title, pageid) = args
    text = filter_wiki(text)
    result = tokenizer_func(text, token_min_len, token_max_len, lower)
    return (result, title, pageid)

def init_to_ignore_interrupt():
    if False:
        while True:
            i = 10
    'Enables interruption ignoring.\n\n    Warnings\n    --------\n    Should only be used when master is prepared to handle termination of\n    child processes.\n\n    '
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _process_article(args):
    if False:
        i = 10
        return i + 15
    'Same as :func:`~gensim.corpora.wikicorpus.process_article`, but with args in list format.\n\n    Parameters\n    ----------\n    args : [(str, bool, str, int), (function, int, int, bool)]\n        First element - same as `args` from :func:`~gensim.corpora.wikicorpus.process_article`,\n        second element is tokenizer function, token minimal length, token maximal length, lowercase flag.\n\n    Returns\n    -------\n    (list of str, str, int)\n        List of tokens from article, title and page id.\n\n    Warnings\n    --------\n    Should not be called explicitly. Use :func:`~gensim.corpora.wikicorpus.process_article` instead.\n\n    '
    (tokenizer_func, token_min_len, token_max_len, lower) = args[-1]
    args = args[:-1]
    return process_article(args, tokenizer_func=tokenizer_func, token_min_len=token_min_len, token_max_len=token_max_len, lower=lower)

class WikiCorpus(TextCorpus):
    """Treat a Wikipedia articles dump as a read-only, streamed, memory-efficient corpus.

    Supported dump formats:

    * <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
    * <LANG>wiki-latest-pages-articles.xml.bz2

    The documents are extracted on-the-fly, so that the whole (massive) dump can stay compressed on disk.

    Notes
    -----
    Dumps for the English Wikipedia can be founded at https://dumps.wikimedia.org/enwiki/.

    Attributes
    ----------
    metadata : bool
        Whether to write articles titles to serialized corpus.

    Warnings
    --------
    "Multistream" archives are *not* supported in Python 2 due to `limitations in the core bz2 library
    <https://docs.python.org/2/library/bz2.html#de-compression-of-files>`_.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import datapath, get_tmpfile
        >>> from gensim.corpora import WikiCorpus, MmCorpus
        >>>
        >>> path_to_wiki_dump = datapath("enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2")
        >>> corpus_path = get_tmpfile("wiki-corpus.mm")
        >>>
        >>> wiki = WikiCorpus(path_to_wiki_dump)  # create word->word_id mapping, ~8h on full wiki
        >>> MmCorpus.serialize(corpus_path, wiki)  # another 8h, creates a file in MatrixMarket format and mapping

    """

    def __init__(self, fname, processes=None, lemmatize=None, dictionary=None, metadata=False, filter_namespaces=('0',), tokenizer_func=tokenize, article_min_tokens=ARTICLE_MIN_WORDS, token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True, filter_articles=None):
        if False:
            i = 10
            return i + 15
        'Initialize the corpus.\n\n        Unless a dictionary is provided, this scans the corpus once,\n        to determine its vocabulary.\n\n        Parameters\n        ----------\n        fname : str\n            Path to the Wikipedia dump file.\n        processes : int, optional\n            Number of processes to run, defaults to `max(1, number of cpu - 1)`.\n        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional\n            Dictionary, if not provided,  this scans the corpus once, to determine its vocabulary\n            **IMPORTANT: this needs a really long time**.\n        filter_namespaces : tuple of str, optional\n            Namespaces to consider.\n        tokenizer_func : function, optional\n            Function that will be used for tokenization. By default, use :func:`~gensim.corpora.wikicorpus.tokenize`.\n            If you inject your own tokenizer, it must conform to this interface:\n            `tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list of str`\n        article_min_tokens : int, optional\n            Minimum tokens in article. Article will be ignored if number of tokens is less.\n        token_min_len : int, optional\n            Minimal token length.\n        token_max_len : int, optional\n            Maximal token length.\n        lower : bool, optional\n             If True - convert all text to lower case.\n        filter_articles: callable or None, optional\n            If set, each XML article element will be passed to this callable before being processed. Only articles\n            where the callable returns an XML element are processed, returning None allows filtering out\n            some articles based on customised rules.\n        metadata: bool\n            Have the `get_texts()` method yield `(content_tokens, (page_id, page_title))` tuples, instead\n            of just `content_tokens`.\n\n        Warnings\n        --------\n        Unless a dictionary is provided, this scans the corpus once, to determine its vocabulary.\n\n        '
        if lemmatize is not None:
            raise NotImplementedError('The lemmatize parameter is no longer supported. If you need to lemmatize, use e.g. <https://github.com/clips/pattern>. Perform lemmatization as part of your tokenization function and pass it as the tokenizer_func parameter to this initializer.')
        self.fname = fname
        self.filter_namespaces = filter_namespaces
        self.filter_articles = filter_articles
        self.metadata = metadata
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes
        self.tokenizer_func = tokenizer_func
        self.article_min_tokens = article_min_tokens
        self.token_min_len = token_min_len
        self.token_max_len = token_max_len
        self.lower = lower
        if dictionary is None:
            self.dictionary = Dictionary(self.get_texts())
        else:
            self.dictionary = dictionary

    @property
    def input(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fname

    def get_texts(self):
        if False:
            i = 10
            return i + 15
        'Iterate over the dump, yielding a list of tokens for each article that passed\n        the length and namespace filtering.\n\n        Uses multiprocessing internally to parallelize the work and process the dump more quickly.\n\n        Notes\n        -----\n        This iterates over the **texts**. If you want vectors, just use the standard corpus interface\n        instead of this method:\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.test.utils import datapath\n            >>> from gensim.corpora import WikiCorpus\n            >>>\n            >>> path_to_wiki_dump = datapath("enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2")\n            >>>\n            >>> for vec in WikiCorpus(path_to_wiki_dump):\n            ...     pass\n\n        Yields\n        ------\n        list of str\n            If `metadata` is False, yield only list of token extracted from the article.\n        (list of str, (int, str))\n            List of tokens (extracted from the article), page id and article title otherwise.\n\n        '
        (articles, articles_all) = (0, 0)
        (positions, positions_all) = (0, 0)
        tokenization_params = (self.tokenizer_func, self.token_min_len, self.token_max_len, self.lower)
        texts = ((text, title, pageid, tokenization_params) for (title, text, pageid) in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces, self.filter_articles))
        pool = multiprocessing.Pool(self.processes, init_to_ignore_interrupt)
        try:
            for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
                for (tokens, title, pageid) in pool.imap(_process_article, group):
                    articles_all += 1
                    positions_all += len(tokens)
                    if len(tokens) < self.article_min_tokens or any((title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES)):
                        continue
                    articles += 1
                    positions += len(tokens)
                    if self.metadata:
                        yield (tokens, (pageid, title))
                    else:
                        yield tokens
        except KeyboardInterrupt:
            logger.warning('user terminated iteration over Wikipedia corpus after %i documents with %i positions (total %i articles, %i positions before pruning articles shorter than %i words)', articles, positions, articles_all, positions_all, self.article_min_tokens)
        except PicklingError as exc:
            raise PicklingError(f'Can not send filtering function {self.filter_articles} to multiprocessing, make sure the function can be pickled.') from exc
        else:
            logger.info('finished iterating over Wikipedia corpus of %i documents with %i positions (total %i articles, %i positions before pruning articles shorter than %i words)', articles, positions, articles_all, positions_all, self.article_min_tokens)
            self.length = articles
        finally:
            pool.terminate()
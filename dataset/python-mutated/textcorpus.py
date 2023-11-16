"""Module provides some code scaffolding to simplify use of built dictionary for constructing BoW vectors.

Notes
-----
Text corpora usually reside on disk, as text files in one format or another In a common scenario,
we need to build a dictionary (a `word->integer id` mapping), which is then used to construct sparse bag-of-word vectors
(= iterable of `(word_id, word_weight)`).

This module provides some code scaffolding to simplify this pipeline. For example, given a corpus where each document
is a separate line in file on disk, you would override the :meth:`gensim.corpora.textcorpus.TextCorpus.get_texts`
to read one line=document at a time, process it (lowercase, tokenize, whatever) and yield it as a sequence of words.

Overriding :meth:`gensim.corpora.textcorpus.TextCorpus.get_texts` is enough, you can then initialize the corpus
with e.g. `MyTextCorpus("mycorpus.txt.bz2")` and it will behave correctly like a corpus of sparse vectors.
The :meth:`~gensim.corpora.textcorpus.TextCorpus.__iter__` method is automatically set up,
and dictionary is automatically populated with all `word->id` mappings.

The resulting object can be used as input to some of gensim models (:class:`~gensim.models.tfidfmodel.TfidfModel`,
:class:`~gensim.models.lsimodel.LsiModel`, :class:`~gensim.models.ldamodel.LdaModel`, ...), serialized with any format
(`Matrix Market <http://math.nist.gov/MatrixMarket/formats.html>`_,
`SvmLight <http://svmlight.joachims.org/>`_, `Blei's LDA-C format <https://github.com/blei-lab/lda-c>`_, etc).


See Also
--------
:class:`gensim.test.test_miislita.CorpusMiislita`
    Good simple example.

"""
from __future__ import with_statement
import logging
import os
import random
import re
import sys
from gensim import interfaces, utils
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import remove_stopword_tokens, remove_short_tokens, lower_to_unicode, strip_multiple_whitespaces
from gensim.utils import deaccent, simple_tokenize
from smart_open import open
logger = logging.getLogger(__name__)

class TextCorpus(interfaces.CorpusABC):
    """Helper class to simplify the pipeline of getting BoW vectors from plain text.

    Notes
    -----
    This is an abstract base class: override the :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` and
    :meth:`~gensim.corpora.textcorpus.TextCorpus.__len__` methods to match your particular input.

    Given a filename (or a file-like object) in constructor, the corpus object will be automatically initialized
    with a dictionary in `self.dictionary` and will support the :meth:`~gensim.corpora.textcorpus.TextCorpus.__iter__`
    corpus method.  You have a few different ways of utilizing this class via subclassing or by construction with
    different preprocessing arguments.

    The :meth:`~gensim.corpora.textcorpus.TextCorpus.__iter__` method converts the lists of tokens produced by
    :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` to BoW format using
    :meth:`gensim.corpora.dictionary.Dictionary.doc2bow`.

    :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` does the following:

    #. Calls :meth:`~gensim.corpora.textcorpus.TextCorpus.getstream` to get a generator over the texts.
       It yields each document in turn from the underlying text file or files.
    #. For each document from the stream, calls :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` to produce
       a list of tokens. If metadata=True, it yields a 2-`tuple` with the document number as the second element.

    Preprocessing consists of 0+ `character_filters`, a `tokenizer`, and 0+ `token_filters`.

    The preprocessing consists of calling each filter in `character_filters` with the document text.
    Unicode is not guaranteed, and if desired, the first filter should convert to unicode.
    The output of each character filter should be another string. The output from the final filter is fed
    to the `tokenizer`, which should split the string into a list of tokens (strings).
    Afterwards, the list of tokens is fed through each filter in `token_filters`. The final output returned from
    :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` is the output from the final token filter.

    So to use this class, you can either pass in different preprocessing functions using the
    `character_filters`, `tokenizer`, and `token_filters` arguments, or you can subclass it.

    If subclassing: override :meth:`~gensim.corpora.textcorpus.TextCorpus.getstream` to take text from different input
    sources in different formats.
    Override :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` if you must provide different initial
    preprocessing, then call the :meth:`~gensim.corpora.textcorpus.TextCorpus.preprocess_text` method to apply
    the normal preprocessing.
    You can also override :meth:`~gensim.corpora.textcorpus.TextCorpus.get_texts` in order to tag the documents
    (token lists) with different metadata.

    The default preprocessing consists of:

    #. :func:`~gensim.parsing.preprocessing.lower_to_unicode` - lowercase and convert to unicode (assumes utf8 encoding)
    #. :func:`~gensim.utils.deaccent`- deaccent (asciifolding)
    #. :func:`~gensim.parsing.preprocessing.strip_multiple_whitespaces` - collapse multiple whitespaces into one
    #. :func:`~gensim.utils.simple_tokenize` - tokenize by splitting on whitespace
    #. :func:`~gensim.parsing.preprocessing.remove_short_tokens` - remove words less than 3 characters long
    #. :func:`~gensim.parsing.preprocessing.remove_stopword_tokens` - remove stopwords

    """

    def __init__(self, input=None, dictionary=None, metadata=False, character_filters=None, tokenizer=None, token_filters=None):
        if False:
            return 10
        "\n\n        Parameters\n        ----------\n        input : str, optional\n            Path to top-level directory (file) to traverse for corpus documents.\n        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional\n            If a dictionary is provided, it will not be updated with the given corpus on initialization.\n            If None - new dictionary will be built for the given corpus.\n            If `input` is None, the dictionary will remain uninitialized.\n        metadata : bool, optional\n            If True - yield metadata with each document.\n        character_filters : iterable of callable, optional\n            Each will be applied to the text of each document in order, and should return a single string with\n            the modified text. For Python 2, the original text will not be unicode, so it may be useful to\n            convert to unicode as the first character filter.\n            If None - using :func:`~gensim.parsing.preprocessing.lower_to_unicode`,\n            :func:`~gensim.utils.deaccent` and :func:`~gensim.parsing.preprocessing.strip_multiple_whitespaces`.\n        tokenizer : callable, optional\n            Tokenizer for document, if None - using :func:`~gensim.utils.simple_tokenize`.\n        token_filters : iterable of callable, optional\n            Each will be applied to the iterable of tokens in order, and should return another iterable of tokens.\n            These filters can add, remove, or replace tokens, or do nothing at all.\n            If None - using :func:`~gensim.parsing.preprocessing.remove_short_tokens` and\n            :func:`~gensim.parsing.preprocessing.remove_stopword_tokens`.\n\n        Examples\n        --------\n        .. sourcecode:: pycon\n\n            >>> from gensim.corpora.textcorpus import TextCorpus\n            >>> from gensim.test.utils import datapath\n            >>> from gensim import utils\n            >>>\n            >>>\n            >>> class CorpusMiislita(TextCorpus):\n            ...     stopwords = set('for a of the and to in on'.split())\n            ...\n            ...     def get_texts(self):\n            ...         for doc in self.getstream():\n            ...             yield [word for word in utils.to_unicode(doc).lower().split() if word not in self.stopwords]\n            ...\n            ...     def __len__(self):\n            ...         self.length = sum(1 for _ in self.get_texts())\n            ...         return self.length\n            >>>\n            >>>\n            >>> corpus = CorpusMiislita(datapath('head500.noblanks.cor.bz2'))\n            >>> len(corpus)\n            250\n            >>> document = next(iter(corpus.get_texts()))\n\n        "
        self.input = input
        self.metadata = metadata
        self.character_filters = character_filters
        if self.character_filters is None:
            self.character_filters = [lower_to_unicode, deaccent, strip_multiple_whitespaces]
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = simple_tokenize
        self.token_filters = token_filters
        if self.token_filters is None:
            self.token_filters = [remove_short_tokens, remove_stopword_tokens]
        self.length = None
        self.dictionary = None
        self.init_dictionary(dictionary)

    def init_dictionary(self, dictionary):
        if False:
            while True:
                i = 10
        'Initialize/update dictionary.\n\n        Parameters\n        ----------\n        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional\n            If a dictionary is provided, it will not be updated with the given corpus on initialization.\n            If None - new dictionary will be built for the given corpus.\n\n        Notes\n        -----\n        If self.input is None - make nothing.\n\n        '
        self.dictionary = dictionary if dictionary is not None else Dictionary()
        if self.input is not None:
            if dictionary is None:
                logger.info('Initializing dictionary')
                metadata_setting = self.metadata
                self.metadata = False
                self.dictionary.add_documents(self.get_texts())
                self.metadata = metadata_setting
            else:
                logger.info('Input stream provided but dictionary already initialized')
        else:
            logger.warning('No input document stream provided; assuming dictionary will be initialized some other way.')

    def __iter__(self):
        if False:
            print('Hello World!')
        'Iterate over the corpus.\n\n        Yields\n        ------\n        list of (int, int)\n            Document in BoW format (+ metadata if self.metadata).\n\n        '
        if self.metadata:
            for (text, metadata) in self.get_texts():
                yield (self.dictionary.doc2bow(text, allow_update=False), metadata)
        else:
            for text in self.get_texts():
                yield self.dictionary.doc2bow(text, allow_update=False)

    def getstream(self):
        if False:
            for i in range(10):
                print('nop')
        'Generate documents from the underlying plain text collection (of one or more files).\n\n        Yields\n        ------\n        str\n            Document read from plain-text file.\n\n        Notes\n        -----\n        After generator end - initialize self.length attribute.\n\n        '
        num_texts = 0
        with utils.file_or_filename(self.input) as f:
            for line in f:
                yield line
                num_texts += 1
        self.length = num_texts

    def preprocess_text(self, text):
        if False:
            i = 10
            return i + 15
        'Apply `self.character_filters`, `self.tokenizer`, `self.token_filters` to a single text document.\n\n        Parameters\n        ---------\n        text : str\n            Document read from plain-text file.\n\n        Return\n        ------\n        list of str\n            List of tokens extracted from `text`.\n\n        '
        for character_filter in self.character_filters:
            text = character_filter(text)
        tokens = self.tokenizer(text)
        for token_filter in self.token_filters:
            tokens = token_filter(tokens)
        return tokens

    def step_through_preprocess(self, text):
        if False:
            print('Hello World!')
        'Apply preprocessor one by one and generate result.\n\n        Warnings\n        --------\n        This is useful for debugging issues with the corpus preprocessing pipeline.\n\n        Parameters\n        ----------\n        text : str\n            Document text read from plain-text file.\n\n        Yields\n        ------\n        (callable, object)\n            Pre-processor, output from pre-processor (based on `text`)\n\n        '
        for character_filter in self.character_filters:
            text = character_filter(text)
            yield (character_filter, text)
        tokens = self.tokenizer(text)
        yield (self.tokenizer, tokens)
        for token_filter in self.token_filters:
            yield (token_filter, token_filter(tokens))

    def get_texts(self):
        if False:
            for i in range(10):
                print('nop')
        'Generate documents from corpus.\n\n        Yields\n        ------\n        list of str\n            Document as sequence of tokens (+ lineno if self.metadata)\n\n        '
        lines = self.getstream()
        if self.metadata:
            for (lineno, line) in enumerate(lines):
                yield (self.preprocess_text(line), (lineno,))
        else:
            for line in lines:
                yield self.preprocess_text(line)

    def sample_texts(self, n, seed=None, length=None):
        if False:
            print('Hello World!')
        'Generate `n` random documents from the corpus without replacement.\n\n        Parameters\n        ----------\n        n : int\n            Number of documents we want to sample.\n        seed : int, optional\n            If specified, use it as a seed for local random generator.\n        length : int, optional\n            Value will used as corpus length (because calculate length of corpus can be costly operation).\n            If not specified - will call `__length__`.\n\n        Raises\n        ------\n        ValueError\n            If `n` less than zero or greater than corpus size.\n\n        Notes\n        -----\n        Given the number of remaining documents in a corpus, we need to choose n elements.\n        The probability for the current element to be chosen is `n` / remaining. If we choose it,  we just decrease\n        the `n` and move to the next element.\n\n        Yields\n        ------\n        list of str\n            Sampled document as sequence of tokens.\n\n        '
        random_generator = random if seed is None else random.Random(seed)
        if length is None:
            length = len(self)
        if not n <= length:
            raise ValueError('n {0:d} is larger/equal than length of corpus {1:d}.'.format(n, length))
        if not 0 <= n:
            raise ValueError('Negative sample size n {0:d}.'.format(n))
        i = 0
        for (i, sample) in enumerate(self.getstream()):
            if i == length:
                break
            remaining_in_corpus = length - i
            chance = random_generator.randint(1, remaining_in_corpus)
            if chance <= n:
                n -= 1
                if self.metadata:
                    yield (self.preprocess_text(sample[0]), sample[1])
                else:
                    yield self.preprocess_text(sample)
        if n != 0:
            raise ValueError('length {0:d} greater than number of documents in corpus {1:d}'.format(length, i + 1))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        'Get length of corpus\n\n        Warnings\n        --------\n        If self.length is None - will read all corpus for calculate this attribute through\n        :meth:`~gensim.corpora.textcorpus.TextCorpus.getstream`.\n\n        Returns\n        -------\n        int\n            Length of corpus.\n\n        '
        if self.length is None:
            self.length = sum((1 for _ in self.getstream()))
        return self.length

class TextDirectoryCorpus(TextCorpus):
    """Read documents recursively from a directory.
    Each file/line (depends on `lines_are_documents`) is interpreted as a plain text document.

    """

    def __init__(self, input, dictionary=None, metadata=False, min_depth=0, max_depth=None, pattern=None, exclude_pattern=None, lines_are_documents=False, encoding='utf-8', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Parameters\n        ----------\n        input : str\n            Path to input file/folder.\n        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional\n            If a dictionary is provided, it will not be updated with the given corpus on initialization.\n            If None - new dictionary will be built for the given corpus.\n            If `input` is None, the dictionary will remain uninitialized.\n        metadata : bool, optional\n            If True - yield metadata with each document.\n        min_depth : int, optional\n            Minimum depth in directory tree at which to begin searching for files.\n        max_depth : int, optional\n            Max depth in directory tree at which files will no longer be considered.\n            If None - not limited.\n        pattern : str, optional\n            Regex to use for file name inclusion, all those files *not* matching this pattern will be ignored.\n        exclude_pattern : str, optional\n            Regex to use for file name exclusion, all files matching this pattern will be ignored.\n        lines_are_documents : bool, optional\n            If True - each line is considered a document, otherwise - each file is one document.\n        encoding : str, optional\n            Encoding used to read the specified file or files in the specified directory.\n        kwargs: keyword arguments passed through to the `TextCorpus` constructor.\n            See :meth:`gemsim.corpora.textcorpus.TextCorpus.__init__` docstring for more details on these.\n\n        '
        self._min_depth = min_depth
        self._max_depth = sys.maxsize if max_depth is None else max_depth
        self.pattern = pattern
        self.exclude_pattern = exclude_pattern
        self.lines_are_documents = lines_are_documents
        self.encoding = encoding
        super(TextDirectoryCorpus, self).__init__(input, dictionary, metadata, **kwargs)

    @property
    def lines_are_documents(self):
        if False:
            i = 10
            return i + 15
        return self._lines_are_documents

    @lines_are_documents.setter
    def lines_are_documents(self, lines_are_documents):
        if False:
            while True:
                i = 10
        self._lines_are_documents = lines_are_documents
        self.length = None

    @property
    def pattern(self):
        if False:
            while True:
                i = 10
        return self._pattern

    @pattern.setter
    def pattern(self, pattern):
        if False:
            while True:
                i = 10
        self._pattern = None if pattern is None else re.compile(pattern)
        self.length = None

    @property
    def exclude_pattern(self):
        if False:
            i = 10
            return i + 15
        return self._exclude_pattern

    @exclude_pattern.setter
    def exclude_pattern(self, pattern):
        if False:
            return 10
        self._exclude_pattern = None if pattern is None else re.compile(pattern)
        self.length = None

    @property
    def min_depth(self):
        if False:
            return 10
        return self._min_depth

    @min_depth.setter
    def min_depth(self, min_depth):
        if False:
            print('Hello World!')
        self._min_depth = min_depth
        self.length = None

    @property
    def max_depth(self):
        if False:
            i = 10
            return i + 15
        return self._max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        if False:
            while True:
                i = 10
        self._max_depth = max_depth
        self.length = None

    def iter_filepaths(self):
        if False:
            i = 10
            return i + 15
        'Generate (lazily)  paths to each file in the directory structure within the specified range of depths.\n        If a filename pattern to match was given, further filter to only those filenames that match.\n\n        Yields\n        ------\n        str\n            Path to file\n\n        '
        for (depth, dirpath, dirnames, filenames) in walk(self.input):
            if self.min_depth <= depth <= self.max_depth:
                if self.pattern is not None:
                    filenames = (n for n in filenames if self.pattern.match(n) is not None)
                if self.exclude_pattern is not None:
                    filenames = (n for n in filenames if self.exclude_pattern.match(n) is None)
                for name in filenames:
                    yield os.path.join(dirpath, name)

    def getstream(self):
        if False:
            print('Hello World!')
        'Generate documents from the underlying plain text collection (of one or more files).\n\n        Yields\n        ------\n        str\n            One document (if lines_are_documents - True), otherwise - each file is one document.\n\n        '
        num_texts = 0
        for path in self.iter_filepaths():
            with open(path, 'rt', encoding=self.encoding) as f:
                if self.lines_are_documents:
                    for line in f:
                        yield line.strip()
                        num_texts += 1
                else:
                    yield f.read().strip()
                    num_texts += 1
        self.length = num_texts

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Get length of corpus.\n\n        Returns\n        -------\n        int\n            Length of corpus.\n\n        '
        if self.length is None:
            self._cache_corpus_length()
        return self.length

    def _cache_corpus_length(self):
        if False:
            while True:
                i = 10
        'Calculate length of corpus and cache it to `self.length`.'
        if not self.lines_are_documents:
            self.length = sum((1 for _ in self.iter_filepaths()))
        else:
            self.length = sum((1 for _ in self.getstream()))

def walk(top, topdown=True, onerror=None, followlinks=False, depth=0):
    if False:
        print('Hello World!')
    "Generate the file names in a directory tree by walking the tree either top-down or bottom-up.\n    For each directory in the tree rooted at directory top (including top itself), it yields a 4-tuple\n    (depth, dirpath, dirnames, filenames).\n\n    Parameters\n    ----------\n    top : str\n        Root directory.\n    topdown : bool, optional\n        If True - you can modify dirnames in-place.\n    onerror : function, optional\n        Some function, will be called with one argument, an OSError instance.\n        It can report the error to continue with the walk, or raise the exception to abort the walk.\n        Note that the filename is available as the filename attribute of the exception object.\n    followlinks : bool, optional\n        If True - visit directories pointed to by symlinks, on systems that support them.\n    depth : int, optional\n        Height of file-tree, don't pass it manually (this used as accumulator for recursion).\n\n    Notes\n    -----\n    This is a mostly copied version of `os.walk` from the Python 2 source code.\n    The only difference is that it returns the depth in the directory tree structure\n    at which each yield is taking place.\n\n    Yields\n    ------\n    (int, str, list of str, list of str)\n        Depth, current path, visited directories, visited non-directories.\n\n    See Also\n    --------\n    `os.walk documentation <https://docs.python.org/2/library/os.html#os.walk>`_\n\n    "
    (islink, join, isdir) = (os.path.islink, os.path.join, os.path.isdir)
    try:
        names = os.listdir(top)
    except OSError as err:
        if onerror is not None:
            onerror(err)
        return
    (dirs, nondirs) = ([], [])
    for name in names:
        if isdir(join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)
    if topdown:
        yield (depth, top, dirs, nondirs)
    for name in dirs:
        new_path = join(top, name)
        if followlinks or not islink(new_path):
            for x in walk(new_path, topdown, onerror, followlinks, depth + 1):
                yield x
    if not topdown:
        yield (depth, top, dirs, nondirs)
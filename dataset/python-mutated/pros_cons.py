"""
CorpusReader for the Pros and Cons dataset.

- Pros and Cons dataset information -

Contact: Bing Liu, liub@cs.uic.edu
        https://www.cs.uic.edu/~liub

Distributed with permission.

Related papers:

- Murthy Ganapathibhotla and Bing Liu. "Mining Opinions in Comparative Sentences".
    Proceedings of the 22nd International Conference on Computational Linguistics
    (Coling-2008), Manchester, 18-22 August, 2008.

- Bing Liu, Minqing Hu and Junsheng Cheng. "Opinion Observer: Analyzing and Comparing
    Opinions on the Web". Proceedings of the 14th international World Wide Web
    conference (WWW-2005), May 10-14, 2005, in Chiba, Japan.
"""
import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *

class ProsConsCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    Reader for the Pros and Cons sentence dataset.

        >>> from nltk.corpus import pros_cons
        >>> pros_cons.sents(categories='Cons') # doctest: +NORMALIZE_WHITESPACE
        [['East', 'batteries', '!', 'On', '-', 'off', 'switch', 'too', 'easy',
        'to', 'maneuver', '.'], ['Eats', '...', 'no', ',', 'GULPS', 'batteries'],
        ...]
        >>> pros_cons.words('IntegratedPros.txt')
        ['Easy', 'to', 'use', ',', 'economical', '!', ...]
    """
    CorpusView = StreamBackedCorpusView

    def __init__(self, root, fileids, word_tokenizer=WordPunctTokenizer(), encoding='utf8', **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        :param root: The root directory for the corpus.\n        :param fileids: a list or regexp specifying the fileids in the corpus.\n        :param word_tokenizer: a tokenizer for breaking sentences or paragraphs\n            into words. Default: `WhitespaceTokenizer`\n        :param encoding: the encoding that should be used to read the corpus.\n        :param kwargs: additional parameters passed to CategorizedCorpusReader.\n        '
        CorpusReader.__init__(self, root, fileids, encoding)
        CategorizedCorpusReader.__init__(self, kwargs)
        self._word_tokenizer = word_tokenizer

    def sents(self, fileids=None, categories=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return all sentences in the corpus or in the specified files/categories.\n\n        :param fileids: a list or regexp specifying the ids of the files whose\n            sentences have to be returned.\n        :param categories: a list specifying the categories whose sentences\n            have to be returned.\n        :return: the given file(s) as a list of sentences. Each sentence is\n            tokenized using the specified word_tokenizer.\n        :rtype: list(list(str))\n        '
        fileids = self._resolve(fileids, categories)
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat([self.CorpusView(path, self._read_sent_block, encoding=enc) for (path, enc, fileid) in self.abspaths(fileids, True, True)])

    def words(self, fileids=None, categories=None):
        if False:
            i = 10
            return i + 15
        '\n        Return all words and punctuation symbols in the corpus or in the specified\n        files/categories.\n\n        :param fileids: a list or regexp specifying the ids of the files whose\n            words have to be returned.\n        :param categories: a list specifying the categories whose words have\n            to be returned.\n        :return: the given file(s) as a list of words and punctuation symbols.\n        :rtype: list(str)\n        '
        fileids = self._resolve(fileids, categories)
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat([self.CorpusView(path, self._read_word_block, encoding=enc) for (path, enc, fileid) in self.abspaths(fileids, True, True)])

    def _read_sent_block(self, stream):
        if False:
            return 10
        sents = []
        for i in range(20):
            line = stream.readline()
            if not line:
                continue
            sent = re.match('^(?!\\n)\\s*<(Pros|Cons)>(.*)</(?:Pros|Cons)>', line)
            if sent:
                sents.append(self._word_tokenizer.tokenize(sent.group(2).strip()))
        return sents

    def _read_word_block(self, stream):
        if False:
            i = 10
            return i + 15
        words = []
        for sent in self._read_sent_block(stream):
            words.extend(sent)
        return words
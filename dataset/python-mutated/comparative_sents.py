"""
CorpusReader for the Comparative Sentence Dataset.

- Comparative Sentence Dataset information -

Annotated by: Nitin Jindal and Bing Liu, 2006.
              Department of Computer Sicence
              University of Illinois at Chicago

Contact: Nitin Jindal, njindal@cs.uic.edu
         Bing Liu, liub@cs.uic.edu (https://www.cs.uic.edu/~liub)

Distributed with permission.

Related papers:

- Nitin Jindal and Bing Liu. "Identifying Comparative Sentences in Text Documents".
   Proceedings of the ACM SIGIR International Conference on Information Retrieval
   (SIGIR-06), 2006.

- Nitin Jindal and Bing Liu. "Mining Comprative Sentences and Relations".
   Proceedings of Twenty First National Conference on Artificial Intelligence
   (AAAI-2006), 2006.

- Murthy Ganapathibhotla and Bing Liu. "Mining Opinions in Comparative Sentences".
    Proceedings of the 22nd International Conference on Computational Linguistics
    (Coling-2008), Manchester, 18-22 August, 2008.
"""
import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
STARS = re.compile('^\\*+$')
COMPARISON = re.compile('<cs-[1234]>')
CLOSE_COMPARISON = re.compile('</cs-[1234]>')
GRAD_COMPARISON = re.compile('<cs-[123]>')
NON_GRAD_COMPARISON = re.compile('<cs-4>')
ENTITIES_FEATS = re.compile('(\\d)_((?:[\\.\\w\\s/-](?!\\d_))+)')
KEYWORD = re.compile('\\(([^\\(]*)\\)$')

class Comparison:
    """
    A Comparison represents a comparative sentence and its constituents.
    """

    def __init__(self, text=None, comp_type=None, entity_1=None, entity_2=None, feature=None, keyword=None):
        if False:
            i = 10
            return i + 15
        '\n        :param text: a string (optionally tokenized) containing a comparison.\n        :param comp_type: an integer defining the type of comparison expressed.\n            Values can be: 1 (Non-equal gradable), 2 (Equative), 3 (Superlative),\n            4 (Non-gradable).\n        :param entity_1: the first entity considered in the comparison relation.\n        :param entity_2: the second entity considered in the comparison relation.\n        :param feature: the feature considered in the comparison relation.\n        :param keyword: the word or phrase which is used for that comparative relation.\n        '
        self.text = text
        self.comp_type = comp_type
        self.entity_1 = entity_1
        self.entity_2 = entity_2
        self.feature = feature
        self.keyword = keyword

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Comparison(text="{}", comp_type={}, entity_1="{}", entity_2="{}", feature="{}", keyword="{}")'.format(self.text, self.comp_type, self.entity_1, self.entity_2, self.feature, self.keyword)

class ComparativeSentencesCorpusReader(CorpusReader):
    """
    Reader for the Comparative Sentence Dataset by Jindal and Liu (2006).

        >>> from nltk.corpus import comparative_sentences
        >>> comparison = comparative_sentences.comparisons()[0]
        >>> comparison.text # doctest: +NORMALIZE_WHITESPACE
        ['its', 'fast-forward', 'and', 'rewind', 'work', 'much', 'more', 'smoothly',
        'and', 'consistently', 'than', 'those', 'of', 'other', 'models', 'i', "'ve",
        'had', '.']
        >>> comparison.entity_2
        'models'
        >>> (comparison.feature, comparison.keyword)
        ('rewind', 'more')
        >>> len(comparative_sentences.comparisons())
        853
    """
    CorpusView = StreamBackedCorpusView

    def __init__(self, root, fileids, word_tokenizer=WhitespaceTokenizer(), sent_tokenizer=None, encoding='utf8'):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param root: The root directory for this corpus.\n        :param fileids: a list or regexp specifying the fileids in this corpus.\n        :param word_tokenizer: tokenizer for breaking sentences or paragraphs\n            into words. Default: `WhitespaceTokenizer`\n        :param sent_tokenizer: tokenizer for breaking paragraphs into sentences.\n        :param encoding: the encoding that should be used to read the corpus.\n        '
        CorpusReader.__init__(self, root, fileids, encoding)
        self._word_tokenizer = word_tokenizer
        self._sent_tokenizer = sent_tokenizer
        self._readme = 'README.txt'

    def comparisons(self, fileids=None):
        if False:
            while True:
                i = 10
        '\n        Return all comparisons in the corpus.\n\n        :param fileids: a list or regexp specifying the ids of the files whose\n            comparisons have to be returned.\n        :return: the given file(s) as a list of Comparison objects.\n        :rtype: list(Comparison)\n        '
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat([self.CorpusView(path, self._read_comparison_block, encoding=enc) for (path, enc, fileid) in self.abspaths(fileids, True, True)])

    def keywords(self, fileids=None):
        if False:
            i = 10
            return i + 15
        '\n        Return a set of all keywords used in the corpus.\n\n        :param fileids: a list or regexp specifying the ids of the files whose\n            keywords have to be returned.\n        :return: the set of keywords and comparative phrases used in the corpus.\n        :rtype: set(str)\n        '
        all_keywords = concat([self.CorpusView(path, self._read_keyword_block, encoding=enc) for (path, enc, fileid) in self.abspaths(fileids, True, True)])
        keywords_set = {keyword.lower() for keyword in all_keywords if keyword}
        return keywords_set

    def keywords_readme(self):
        if False:
            while True:
                i = 10
        '\n        Return the list of words and constituents considered as clues of a\n        comparison (from listOfkeywords.txt).\n        '
        keywords = []
        with self.open('listOfkeywords.txt') as fp:
            raw_text = fp.read()
        for line in raw_text.split('\n'):
            if not line or line.startswith('//'):
                continue
            keywords.append(line.strip())
        return keywords

    def sents(self, fileids=None):
        if False:
            return 10
        '\n        Return all sentences in the corpus.\n\n        :param fileids: a list or regexp specifying the ids of the files whose\n            sentences have to be returned.\n        :return: all sentences of the corpus as lists of tokens (or as plain\n            strings, if no word tokenizer is specified).\n        :rtype: list(list(str)) or list(str)\n        '
        return concat([self.CorpusView(path, self._read_sent_block, encoding=enc) for (path, enc, fileid) in self.abspaths(fileids, True, True)])

    def words(self, fileids=None):
        if False:
            return 10
        '\n        Return all words and punctuation symbols in the corpus.\n\n        :param fileids: a list or regexp specifying the ids of the files whose\n            words have to be returned.\n        :return: the given file(s) as a list of words and punctuation symbols.\n        :rtype: list(str)\n        '
        return concat([self.CorpusView(path, self._read_word_block, encoding=enc) for (path, enc, fileid) in self.abspaths(fileids, True, True)])

    def _read_comparison_block(self, stream):
        if False:
            print('Hello World!')
        while True:
            line = stream.readline()
            if not line:
                return []
            comparison_tags = re.findall(COMPARISON, line)
            if comparison_tags:
                grad_comparisons = re.findall(GRAD_COMPARISON, line)
                non_grad_comparisons = re.findall(NON_GRAD_COMPARISON, line)
                comparison_text = stream.readline().strip()
                if self._word_tokenizer:
                    comparison_text = self._word_tokenizer.tokenize(comparison_text)
                stream.readline()
                comparison_bundle = []
                if grad_comparisons:
                    for comp in grad_comparisons:
                        comp_type = int(re.match('<cs-(\\d)>', comp).group(1))
                        comparison = Comparison(text=comparison_text, comp_type=comp_type)
                        line = stream.readline()
                        entities_feats = ENTITIES_FEATS.findall(line)
                        if entities_feats:
                            for (code, entity_feat) in entities_feats:
                                if code == '1':
                                    comparison.entity_1 = entity_feat.strip()
                                elif code == '2':
                                    comparison.entity_2 = entity_feat.strip()
                                elif code == '3':
                                    comparison.feature = entity_feat.strip()
                        keyword = KEYWORD.findall(line)
                        if keyword:
                            comparison.keyword = keyword[0]
                        comparison_bundle.append(comparison)
                if non_grad_comparisons:
                    for comp in non_grad_comparisons:
                        comp_type = int(re.match('<cs-(\\d)>', comp).group(1))
                        comparison = Comparison(text=comparison_text, comp_type=comp_type)
                        comparison_bundle.append(comparison)
                return comparison_bundle

    def _read_keyword_block(self, stream):
        if False:
            i = 10
            return i + 15
        keywords = []
        for comparison in self._read_comparison_block(stream):
            keywords.append(comparison.keyword)
        return keywords

    def _read_sent_block(self, stream):
        if False:
            return 10
        while True:
            line = stream.readline()
            if re.match(STARS, line):
                while True:
                    line = stream.readline()
                    if re.match(STARS, line):
                        break
                continue
            if not re.findall(COMPARISON, line) and (not ENTITIES_FEATS.findall(line)) and (not re.findall(CLOSE_COMPARISON, line)):
                if self._sent_tokenizer:
                    return [self._word_tokenizer.tokenize(sent) for sent in self._sent_tokenizer.tokenize(line)]
                else:
                    return [self._word_tokenizer.tokenize(line)]

    def _read_word_block(self, stream):
        if False:
            i = 10
            return i + 15
        words = []
        for sent in self._read_sent_block(stream):
            words.extend(sent)
        return words
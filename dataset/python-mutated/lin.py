import re
from collections import defaultdict
from functools import reduce
from nltk.corpus.reader import CorpusReader

class LinThesaurusCorpusReader(CorpusReader):
    """Wrapper for the LISP-formatted thesauruses distributed by Dekang Lin."""
    _key_re = re.compile('\\("?([^"]+)"? \\(desc [0-9.]+\\).+')

    @staticmethod
    def __defaultdict_factory():
        if False:
            for i in range(10):
                print('nop')
        'Factory for creating defaultdict of defaultdict(dict)s'
        return defaultdict(dict)

    def __init__(self, root, badscore=0.0):
        if False:
            for i in range(10):
                print('nop')
        "\n        Initialize the thesaurus.\n\n        :param root: root directory containing thesaurus LISP files\n        :type root: C{string}\n        :param badscore: the score to give to words which do not appear in each other's sets of synonyms\n        :type badscore: C{float}\n        "
        super().__init__(root, 'sim[A-Z]\\.lsp')
        self._thesaurus = defaultdict(LinThesaurusCorpusReader.__defaultdict_factory)
        self._badscore = badscore
        for (path, encoding, fileid) in self.abspaths(include_encoding=True, include_fileid=True):
            with open(path) as lin_file:
                first = True
                for line in lin_file:
                    line = line.strip()
                    if first:
                        key = LinThesaurusCorpusReader._key_re.sub('\\1', line)
                        first = False
                    elif line == '))':
                        first = True
                    else:
                        split_line = line.split('\t')
                        if len(split_line) == 2:
                            (ngram, score) = split_line
                            self._thesaurus[fileid][key][ngram.strip('"')] = float(score)

    def similarity(self, ngram1, ngram2, fileid=None):
        if False:
            print('Hello World!')
        '\n        Returns the similarity score for two ngrams.\n\n        :param ngram1: first ngram to compare\n        :type ngram1: C{string}\n        :param ngram2: second ngram to compare\n        :type ngram2: C{string}\n        :param fileid: thesaurus fileid to search in. If None, search all fileids.\n        :type fileid: C{string}\n        :return: If fileid is specified, just the score for the two ngrams; otherwise,\n                 list of tuples of fileids and scores.\n        '
        if ngram1 == ngram2:
            if fileid:
                return 1.0
            else:
                return [(fid, 1.0) for fid in self._fileids]
        elif fileid:
            return self._thesaurus[fileid][ngram1][ngram2] if ngram2 in self._thesaurus[fileid][ngram1] else self._badscore
        else:
            return [(fid, self._thesaurus[fid][ngram1][ngram2] if ngram2 in self._thesaurus[fid][ngram1] else self._badscore) for fid in self._fileids]

    def scored_synonyms(self, ngram, fileid=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a list of scored synonyms (tuples of synonyms and scores) for the current ngram\n\n        :param ngram: ngram to lookup\n        :type ngram: C{string}\n        :param fileid: thesaurus fileid to search in. If None, search all fileids.\n        :type fileid: C{string}\n        :return: If fileid is specified, list of tuples of scores and synonyms; otherwise,\n                 list of tuples of fileids and lists, where inner lists consist of tuples of\n                 scores and synonyms.\n        '
        if fileid:
            return self._thesaurus[fileid][ngram].items()
        else:
            return [(fileid, self._thesaurus[fileid][ngram].items()) for fileid in self._fileids]

    def synonyms(self, ngram, fileid=None):
        if False:
            while True:
                i = 10
        '\n        Returns a list of synonyms for the current ngram.\n\n        :param ngram: ngram to lookup\n        :type ngram: C{string}\n        :param fileid: thesaurus fileid to search in. If None, search all fileids.\n        :type fileid: C{string}\n        :return: If fileid is specified, list of synonyms; otherwise, list of tuples of fileids and\n                 lists, where inner lists contain synonyms.\n        '
        if fileid:
            return self._thesaurus[fileid][ngram].keys()
        else:
            return [(fileid, self._thesaurus[fileid][ngram].keys()) for fileid in self._fileids]

    def __contains__(self, ngram):
        if False:
            return 10
        '\n        Determines whether or not the given ngram is in the thesaurus.\n\n        :param ngram: ngram to lookup\n        :type ngram: C{string}\n        :return: whether the given ngram is in the thesaurus.\n        '
        return reduce(lambda accum, fileid: accum or ngram in self._thesaurus[fileid], self._fileids, False)

def demo():
    if False:
        while True:
            i = 10
    from nltk.corpus import lin_thesaurus as thes
    word1 = 'business'
    word2 = 'enterprise'
    print('Getting synonyms for ' + word1)
    print(thes.synonyms(word1))
    print('Getting scored synonyms for ' + word1)
    print(thes.scored_synonyms(word1))
    print('Getting synonyms from simN.lsp (noun subsection) for ' + word1)
    print(thes.synonyms(word1, fileid='simN.lsp'))
    print('Getting synonyms from simN.lsp (noun subsection) for ' + word1)
    print(thes.synonyms(word1, fileid='simN.lsp'))
    print(f'Similarity score for {word1} and {word2}:')
    print(thes.similarity(word1, word2))
if __name__ == '__main__':
    demo()
import re
from collections import defaultdict, namedtuple
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.wordlist import WordListCorpusReader
from nltk.tokenize import line_tokenize
PanlexLanguage = namedtuple('PanlexLanguage', ['panlex_uid', 'iso639', 'iso639_type', 'script', 'name', 'langvar_uid'])

class PanlexSwadeshCorpusReader(WordListCorpusReader):
    """
    This is a class to read the PanLex Swadesh list from

    David Kamholz, Jonathan Pool, and Susan M. Colowick (2014).
    PanLex: Building a Resource for Panlingual Lexical Translation.
    In LREC. http://www.lrec-conf.org/proceedings/lrec2014/pdf/1029_Paper.pdf

    License: CC0 1.0 Universal
    https://creativecommons.org/publicdomain/zero/1.0/legalcode
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.swadesh_size = re.match('swadesh([0-9].*)\\/', self.fileids()[0]).group(1)
        self._languages = {lang.panlex_uid: lang for lang in self.get_languages()}
        self._macro_langauges = self.get_macrolanguages()

    def license(self):
        if False:
            for i in range(10):
                print('nop')
        return 'CC0 1.0 Universal'

    def language_codes(self):
        if False:
            i = 10
            return i + 15
        return self._languages.keys()

    def get_languages(self):
        if False:
            return 10
        for line in self.raw(f'langs{self.swadesh_size}.txt').split('\n'):
            if not line.strip():
                continue
            yield PanlexLanguage(*line.strip().split('\t'))

    def get_macrolanguages(self):
        if False:
            return 10
        macro_langauges = defaultdict(list)
        for lang in self._languages.values():
            macro_langauges[lang.iso639].append(lang.panlex_uid)
        return macro_langauges

    def words_by_lang(self, lang_code):
        if False:
            i = 10
            return i + 15
        '\n        :return: a list of list(str)\n        '
        fileid = f'swadesh{self.swadesh_size}/{lang_code}.txt'
        return [concept.split('\t') for concept in self.words(fileid)]

    def words_by_iso639(self, iso63_code):
        if False:
            i = 10
            return i + 15
        '\n        :return: a list of list(str)\n        '
        fileids = [f'swadesh{self.swadesh_size}/{lang_code}.txt' for lang_code in self._macro_langauges[iso63_code]]
        return [concept.split('\t') for fileid in fileids for concept in self.words(fileid)]

    def entries(self, fileids=None):
        if False:
            while True:
                i = 10
        '\n        :return: a tuple of words for the specified fileids.\n        '
        if not fileids:
            fileids = self.fileids()
        wordlists = [self.words(f) for f in fileids]
        return list(zip(*wordlists))
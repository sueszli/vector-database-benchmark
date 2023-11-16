from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from collections import namedtuple
from operator import attrgetter
from ..utils import ItemsCount
from .._compat import to_unicode
from ..nlp.stemmers import null_stemmer
SentenceInfo = namedtuple('SentenceInfo', ('sentence', 'order', 'rating'))

class AbstractSummarizer(object):

    def __init__(self, stemmer=null_stemmer):
        if False:
            while True:
                i = 10
        if not callable(stemmer):
            raise ValueError('Stemmer has to be a callable object')
        self._stemmer = stemmer

    def __call__(self, document, sentences_count):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('This method should be overriden in subclass')

    def stem_word(self, word):
        if False:
            for i in range(10):
                print('nop')
        return self._stemmer(self.normalize_word(word))

    @staticmethod
    def normalize_word(word):
        if False:
            for i in range(10):
                print('nop')
        return to_unicode(word).lower()

    @staticmethod
    def _get_best_sentences(sentences, count, rating, *args, **kwargs):
        if False:
            while True:
                i = 10
        rate = rating
        if isinstance(rating, dict):
            assert not args and (not kwargs)

            def rate(s):
                if False:
                    while True:
                        i = 10
                return rating[s]
        infos = (SentenceInfo(s, o, rate(s, *args, **kwargs)) for (o, s) in enumerate(sentences))
        infos = sorted(infos, key=attrgetter('rating'), reverse=True)
        if not callable(count):
            count = ItemsCount(count)
        infos = count(infos)
        infos = sorted(infos, key=attrgetter('order'))
        return tuple((i.sentence for i in infos))
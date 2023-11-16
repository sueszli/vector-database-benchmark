from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from itertools import chain
from ..._compat import unicode_compatible
from ...utils import cached_property
from ._sentence import Sentence

@unicode_compatible
class Paragraph(object):
    __slots__ = ('_sentences', '_cached_property_sentences', '_cached_property_headings', '_cached_property_words')

    def __init__(self, sentences):
        if False:
            return 10
        sentences = tuple(sentences)
        for sentence in sentences:
            if not isinstance(sentence, Sentence):
                raise TypeError("Only instances of class 'Sentence' are allowed.")
        self._sentences = sentences

    @cached_property
    def sentences(self):
        if False:
            i = 10
            return i + 15
        return tuple((s for s in self._sentences if not s.is_heading))

    @cached_property
    def headings(self):
        if False:
            while True:
                i = 10
        return tuple((s for s in self._sentences if s.is_heading))

    @cached_property
    def words(self):
        if False:
            return 10
        return tuple(chain(*(s.words for s in self._sentences)))

    def __unicode__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<Paragraph with %d headings & %d sentences>' % (len(self.headings), len(self.sentences))

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__str__()
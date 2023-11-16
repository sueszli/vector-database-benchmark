from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from itertools import chain
from ...utils import cached_property
from ..._compat import unicode_compatible

@unicode_compatible
class ObjectDocumentModel(object):

    def __init__(self, paragraphs):
        if False:
            print('Hello World!')
        self._paragraphs = tuple(paragraphs)

    @property
    def paragraphs(self):
        if False:
            while True:
                i = 10
        return self._paragraphs

    @cached_property
    def sentences(self):
        if False:
            while True:
                i = 10
        sentences = (p.sentences for p in self._paragraphs)
        return tuple(chain(*sentences))

    @cached_property
    def headings(self):
        if False:
            return 10
        headings = (p.headings for p in self._paragraphs)
        return tuple(chain(*headings))

    @cached_property
    def words(self):
        if False:
            print('Hello World!')
        words = (p.words for p in self._paragraphs)
        return tuple(chain(*words))

    def __unicode__(self):
        if False:
            return 10
        return '<DOM with %d paragraphs>' % len(self.paragraphs)

    def __repr__(self):
        if False:
            return 10
        return self.__str__()
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from ...utils import cached_property
from ..._compat import to_unicode, to_string, unicode_compatible

@unicode_compatible
class Sentence(object):
    __slots__ = ('_text', '_cached_property_words', '_tokenizer', '_is_heading')

    def __init__(self, text, tokenizer, is_heading=False):
        if False:
            print('Hello World!')
        self._text = to_unicode(text).strip()
        self._tokenizer = tokenizer
        self._is_heading = bool(is_heading)

    @cached_property
    def words(self):
        if False:
            print('Hello World!')
        return self._tokenizer.to_words(self._text)

    @property
    def is_heading(self):
        if False:
            while True:
                i = 10
        return self._is_heading

    def __eq__(self, sentence):
        if False:
            return 10
        assert isinstance(sentence, Sentence)
        return self._is_heading is sentence._is_heading and self._text == sentence._text

    def __ne__(self, sentence):
        if False:
            print('Hello World!')
        return not self.__eq__(sentence)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self._is_heading, self._text))

    def __unicode__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._text

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return to_string('<%s: %s>') % ('Heading' if self._is_heading else 'Sentence', self.__str__())
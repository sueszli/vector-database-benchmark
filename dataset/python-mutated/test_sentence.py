"""
Tests for positioning sentences.
"""
import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
sentinelValueOne = 'someStringValue'
sentinelValueTwo = 'someOtherStringValue'

class DummyProtocol:
    """
    A simple, fake protocol.
    """

    @staticmethod
    def getSentenceAttributes():
        if False:
            while True:
                i = 10
        return ['type', sentinelValueOne, sentinelValueTwo]

class DummySentence(_sentence._BaseSentence):
    """
    A sentence for L{DummyProtocol}.
    """
    ALLOWED_ATTRIBUTES = DummyProtocol.getSentenceAttributes()

class MixinProtocol(_sentence._PositioningSentenceProducerMixin):
    """
    A simple, fake protocol that declaratively tells you the sentences
    it produces using L{base.PositioningSentenceProducerMixin}.
    """
    _SENTENCE_CONTENTS = {None: [sentinelValueOne, sentinelValueTwo, None]}

class MixinSentence(_sentence._BaseSentence):
    """
    A sentence for L{MixinProtocol}.
    """
    ALLOWED_ATTRIBUTES = MixinProtocol.getSentenceAttributes()

class SentenceTestsMixin:
    """
    Tests for positioning protocols and their respective sentences.
    """

    def test_attributeAccess(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A sentence attribute gets the correct value, and accessing an\n        unset attribute (which is specified as being a valid sentence\n        attribute) gets L{None}.\n        '
        thisSentinel = object()
        sentence = self.sentenceClass({sentinelValueOne: thisSentinel})
        self.assertEqual(getattr(sentence, sentinelValueOne), thisSentinel)
        self.assertIsNone(getattr(sentence, sentinelValueTwo))

    def test_raiseOnMissingAttributeAccess(self):
        if False:
            while True:
                i = 10
        '\n        Accessing a nonexistent attribute raises C{AttributeError}.\n        '
        sentence = self.sentenceClass({})
        self.assertRaises(AttributeError, getattr, sentence, 'BOGUS')

    def test_raiseOnBadAttributeAccess(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Accessing bogus attributes raises C{AttributeError}, *even*\n        when that attribute actually is in the sentence data.\n        '
        sentence = self.sentenceClass({'BOGUS': None})
        self.assertRaises(AttributeError, getattr, sentence, 'BOGUS')
    sentenceType = 'tummies'
    reprTemplate = '<%s (%s) {%s}>'

    def _expectedRepr(self, sentenceType='unknown type', dataRepr=''):
        if False:
            i = 10
            return i + 15
        '\n        Builds the expected repr for a sentence.\n\n        @param sentenceType: The name of the sentence type (e.g "GPGGA").\n        @type sentenceType: C{str}\n        @param dataRepr: The repr of the data in the sentence.\n        @type dataRepr: C{str}\n        @return: The expected repr of the sentence.\n        @rtype: C{str}\n        '
        clsName = self.sentenceClass.__name__
        return self.reprTemplate % (clsName, sentenceType, dataRepr)

    def test_unknownTypeRepr(self):
        if False:
            while True:
                i = 10
        '\n        Test the repr of an empty sentence of unknown type.\n        '
        sentence = self.sentenceClass({})
        expectedRepr = self._expectedRepr()
        self.assertEqual(repr(sentence), expectedRepr)

    def test_knownTypeRepr(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the repr of an empty sentence of known type.\n        '
        sentence = self.sentenceClass({'type': self.sentenceType})
        expectedRepr = self._expectedRepr(self.sentenceType)
        self.assertEqual(repr(sentence), expectedRepr)

class MixinTests(TestCase, SentenceTestsMixin):
    """
    Tests for protocols deriving from L{base.PositioningSentenceProducerMixin}
    and their sentences.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.protocol = MixinProtocol()
        self.sentenceClass = MixinSentence

    def test_noNoneInSentenceAttributes(self):
        if False:
            return 10
        '\n        L{None} does not appear in the sentence attributes of the\n        protocol, even though it\'s in the specification.\n\n        This is because L{None} is a placeholder for parts of the sentence you\n        don\'t really need or want, but there are some bits later on in the\n        sentence that you do want. The alternative would be to have to specify\n        things like "_UNUSED0", "_UNUSED1"... which would end up cluttering\n        the sentence data and eventually adapter state.\n        '
        sentenceAttributes = self.protocol.getSentenceAttributes()
        self.assertNotIn(None, sentenceAttributes)
        sentenceContents = self.protocol._SENTENCE_CONTENTS
        sentenceSpecAttributes = itertools.chain(*sentenceContents.values())
        self.assertIn(None, sentenceSpecAttributes)
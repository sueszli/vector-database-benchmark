from __future__ import unicode_literals
import os
import unittest
from nose.tools import *
from nose.plugins.attrib import attr
from textblob.base import BaseTagger
import textblob.taggers
HERE = os.path.abspath(os.path.dirname(__file__))
AP_MODEL_LOC = os.path.join(HERE, 'trontagger.pickle')

class TestPatternTagger(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.text = 'Simple is better than complex. Complex is better than complicated.'
        self.tagger = textblob.taggers.PatternTagger()

    def test_init(self):
        if False:
            return 10
        tagger = textblob.taggers.PatternTagger()
        assert_true(isinstance(tagger, textblob.taggers.BaseTagger))

    def test_tag(self):
        if False:
            return 10
        tags = self.tagger.tag(self.text)
        assert_equal(tags, [('Simple', 'JJ'), ('is', 'VBZ'), ('better', 'JJR'), ('than', 'IN'), ('complex', 'JJ'), ('.', '.'), ('Complex', 'NNP'), ('is', 'VBZ'), ('better', 'JJR'), ('than', 'IN'), ('complicated', 'VBN'), ('.', '.')])

@attr('slow')
@attr('no_pypy')
@attr('requires_numpy')
class TestNLTKTagger(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.text = 'Simple is better than complex. Complex is better than complicated.'
        self.tagger = textblob.taggers.NLTKTagger()

    def test_tag(self):
        if False:
            i = 10
            return i + 15
        tags = self.tagger.tag(self.text)
        assert_equal(tags, [('Simple', 'NN'), ('is', 'VBZ'), ('better', 'JJR'), ('than', 'IN'), ('complex', 'JJ'), ('.', '.'), ('Complex', 'NNP'), ('is', 'VBZ'), ('better', 'JJR'), ('than', 'IN'), ('complicated', 'VBN'), ('.', '.')])

def test_cannot_instantiate_incomplete_tagger():
    if False:
        for i in range(10):
            print('nop')

    class BadTagger(BaseTagger):
        """A tagger without a tag method. How useless."""
        pass
    assert_raises(TypeError, lambda : BadTagger())
if __name__ == '__main__':
    unittest.main()
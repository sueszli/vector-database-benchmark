import unittest
import dedupe

class ParsingTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.index = dedupe.tfidf.TfIdfIndex()

    def test_keywords(self):
        if False:
            return 10
        self.index.index(('AND', 'OR', 'EOF', 'NOT'))
        self.index._index.initSearch()
        assert self.index.search(('AND', 'OR', 'EOF', 'NOT'))[0] == 1

    def test_keywords_title(self):
        if False:
            i = 10
            return i + 15
        self.index.index(('And', 'Or', 'Eof', 'Not'))
        self.index._index.initSearch()
        assert self.index.search(('And', 'Or', 'Eof', 'Not'))[0] == 1

    def test_empty_search(self):
        if False:
            print('Hello World!')
        self.index._index.initSearch()
        assert self.index.search(()) == []

    def test_wildcards(self):
        if False:
            print('Hello World!')
        self.index.index(('f\\o',))
        self.index.index(('f*',))
        self.index._index.initSearch()
        assert len(self.index.search(('f*',))) == 1
if __name__ == '__main__':
    unittest.main()
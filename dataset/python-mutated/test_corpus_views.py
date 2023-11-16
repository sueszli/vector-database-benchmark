"""
Corpus View Regression Tests
"""
import unittest
import nltk.data
from nltk.corpus.reader.util import StreamBackedCorpusView, read_line_block, read_whitespace_block

class TestCorpusViews(unittest.TestCase):
    linetok = nltk.LineTokenizer(blanklines='keep')
    names = ['corpora/inaugural/README', 'corpora/inaugural/1793-Washington.txt', 'corpora/inaugural/1909-Taft.txt']

    def data(self):
        if False:
            for i in range(10):
                print('nop')
        for name in self.names:
            f = nltk.data.find(name)
            with f.open() as fp:
                file_data = fp.read().decode('utf8')
            yield (f, file_data)

    def test_correct_values(self):
        if False:
            return 10
        for (f, file_data) in self.data():
            v = StreamBackedCorpusView(f, read_whitespace_block)
            self.assertEqual(list(v), file_data.split())
            v = StreamBackedCorpusView(f, read_line_block)
            self.assertEqual(list(v), self.linetok.tokenize(file_data))

    def test_correct_length(self):
        if False:
            for i in range(10):
                print('nop')
        for (f, file_data) in self.data():
            v = StreamBackedCorpusView(f, read_whitespace_block)
            self.assertEqual(len(v), len(file_data.split()))
            v = StreamBackedCorpusView(f, read_line_block)
            self.assertEqual(len(v), len(self.linetok.tokenize(file_data)))
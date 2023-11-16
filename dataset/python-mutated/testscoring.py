"""
Scoring module tests
"""
import os
import tempfile
import unittest
from unittest.mock import patch
from txtai.scoring import ScoringFactory

class TestScoring(unittest.TestCase):
    """
    Scoring tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize test data.\n        '
        cls.data = ['US tops 5 million confirmed virus cases', "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", 'Beijing mobilises invasion craft along coast as Taiwan tensions escalate', 'The National Park Service warns against sacrificing slower friends in a bear attack', 'Maine man wins $1M from $25 lottery ticket', 'wins wins wins', 'Make huge profits without work, earn up to $100,000 a day']
        cls.data = [(uid, x, None) for (uid, x) in enumerate(cls.data)]

    def testBM25(self):
        if False:
            return 10
        '\n        Test bm25\n        '
        self.runTests('bm25')

    def testCustom(self):
        if False:
            i = 10
            return i + 15
        '\n        Test custom method\n        '
        self.runTests('txtai.scoring.BM25')

    def testCustomNotFound(self):
        if False:
            while True:
                i = 10
        '\n        Test unresolvable custom method\n        '
        with self.assertRaises(ImportError):
            ScoringFactory.create('notfound.scoring')

    def testSIF(self):
        if False:
            i = 10
            return i + 15
        '\n        Test sif\n        '
        self.runTests('sif')

    def testTFIDF(self):
        if False:
            while True:
                i = 10
        '\n        Test tfidf\n        '
        self.runTests('tfidf')

    def runTests(self, method):
        if False:
            for i in range(10):
                print('nop')
        '\n        Runs a series of tests for a scoring method.\n\n        Args:\n            method: scoring method\n        '
        config = {'method': method}
        self.index(config)
        self.upsert(config)
        self.weights(config)
        self.search(config)
        self.delete(config)
        self.normalize(config)
        self.content(config)
        self.empty(config)
        self.copy(config)
        self.settings(config)

    def index(self, config, data=None):
        if False:
            return 10
        '\n        Test scoring index method.\n\n        Args:\n            config: scoring config\n            data: data to index with scoring method\n\n        Returns:\n            scoring\n        '
        data = data if data else self.data
        scoring = ScoringFactory.create(config)
        scoring.index(data)
        keys = [k for (k, v) in sorted(scoring.idf.items(), key=lambda x: x[1])]
        self.assertEqual(scoring.count(), len(data))
        self.assertEqual(keys[0], 'wins')
        self.assertIsNotNone(self.save(scoring, config, f"scoring.{config['method']}.index"))
        self.assertIsNone(scoring.search('query'))
        return scoring

    def upsert(self, config):
        if False:
            while True:
                i = 10
        '\n        Test scoring upsert method\n        '
        scoring = ScoringFactory.create({**config, **{'tokenizer': {'alphanum': True, 'stopwords': True}}})
        scoring.upsert(self.data)
        self.assertEqual(scoring.count(), len(self.data))
        self.assertFalse('and' in scoring.idf)

    def save(self, scoring, config, name):
        if False:
            i = 10
            return i + 15
        '\n        Test scoring index save/load.\n\n        Args:\n            scoring: scoring index\n            config: scoring config\n            name: output file name\n\n        Returns:\n            scoring\n        '
        index = os.path.join(tempfile.gettempdir(), 'scoring')
        os.makedirs(index, exist_ok=True)
        scoring.save(f'{index}/{name}')
        scoring = ScoringFactory.create(config)
        scoring.load(f'{index}/{name}')
        return scoring

    def weights(self, config):
        if False:
            i = 10
            return i + 15
        '\n        Test standard and tag weighted scores.\n\n        Args:\n            config: scoring config\n        '
        document = (1, ['bear', 'wins'], None)
        scoring = self.index(config)
        weights = scoring.weights(document[1])
        self.assertNotEqual(weights[0], weights[1])
        data = self.data[:]
        (uid, text, _) = data[3]
        data[3] = (uid, text, 'wins')
        scoring = self.index(config, data)
        weights = scoring.weights(document[1])
        self.assertEqual(weights[0], weights[1])

    def search(self, config):
        if False:
            i = 10
            return i + 15
        '\n        Test scoring search.\n\n        Args:\n            config: scoring config\n        '
        config = {**config, **{'terms': True}}
        scoring = ScoringFactory.create(config)
        scoring.index(self.data)
        (index, _) = scoring.search('bear', 1)[0]
        self.assertEqual(index, 3)
        (index, _) = scoring.batchsearch(['bear'], 1)[0][0]
        self.assertEqual(index, 3)
        self.save(scoring, config, f"scoring.{config['method']}.search")
        (index, _) = scoring.search('bear', 1)[0]
        self.assertEqual(index, 3)

    def delete(self, config):
        if False:
            print('Hello World!')
        '\n        Test delete.\n        '
        config = {**config, **{'terms': True, 'content': True}}
        scoring = ScoringFactory.create(config)
        scoring.index(self.data)
        index = scoring.search('bear', 1)[0]['id']
        scoring.delete([index])
        self.assertFalse(scoring.search('bear', 1))
        self.save(scoring, config, f"scoring.{config['method']}.delete")
        self.assertEqual(scoring.count(), len(self.data) - 1)

    def normalize(self, config):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test scoring search with normalized scores.\n\n        Args:\n            method: scoring method\n        '
        scoring = ScoringFactory.create({**config, **{'terms': True, 'normalize': True}})
        scoring.index(self.data)
        (index, score) = scoring.search(self.data[3][1], 1)[0]
        self.assertEqual(index, 3)
        self.assertEqual(score, 1.0)

    def content(self, config):
        if False:
            i = 10
            return i + 15
        '\n        Test scoring search with content.\n\n        Args:\n            config: scoring config\n        '
        scoring = ScoringFactory.create({**config, **{'terms': True, 'content': True}})
        scoring.index(self.data)
        text = 'Great news today'
        scoring.index([(scoring.total, text, None)])
        result = scoring.search('great news', 1)[0]['text']
        self.assertEqual(result, text)
        text = 'Feel good story: baby panda born'
        scoring.index([(scoring.total, {'text': text}, None)])
        result = scoring.search('feel good story', 1)[0]['text']
        self.assertEqual(result, text)

    def empty(self, config):
        if False:
            i = 10
            return i + 15
        '\n        Test scoring index properly handles an index call when no data present.\n\n        Args:\n            config: scoring config\n        '
        scoring = ScoringFactory.create(config)
        scoring.index([])
        self.assertEqual(scoring.total, 0)

    def copy(self, config):
        if False:
            i = 10
            return i + 15
        '\n        Test scoring index copy method.\n        '
        scoring = ScoringFactory.create({**config, **{'terms': True}})
        scoring.index(self.data)
        index = os.path.join(tempfile.gettempdir(), 'scoring')
        os.makedirs(index, exist_ok=True)
        path = f"{index}/scoring.{config['method']}.copy"
        with open(f'{index}.terms', 'w', encoding='utf-8') as f:
            f.write('TEST')
        scoring.save(path)
        self.assertTrue(os.path.exists(path))

    @patch('sys.byteorder', 'big')
    def settings(self, config):
        if False:
            return 10
        '\n        Tests various settings.\n\n        Args:\n            config: scoring config\n        '
        config = {**config, **{'terms': {'cachelimit': 0, 'cutoff': 0.25, 'wal': True}}}
        scoring = ScoringFactory.create(config)
        scoring.index(self.data)
        self.save(scoring, config, f"scoring.{config['method']}.settings")
        (index, _) = scoring.search('bear bear bear wins', 1)[0]
        self.assertEqual(index, 3)
        self.save(scoring, config, f"scoring.{config['method']}.settings")
        self.save(scoring, config, f"scoring.{config['method']}.move")
        self.assertEqual(scoring.count(), len(self.data))
"""
Extractor module tests
"""
import platform
import unittest
from txtai.embeddings import Embeddings
from txtai.pipeline import Extractor, Questions, Similarity

class TestExtractor(unittest.TestCase):
    """
    Extractor tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        '\n        Create single extractor instance.\n        '
        cls.data = ['Giants hit 3 HRs to down Dodgers', 'Giants 5 Dodgers 4 final', 'Dodgers drop Game 2 against the Giants, 5-4', 'Blue Jays beat Red Sox final score 2-1', 'Red Sox lost to the Blue Jays, 2-1', 'Blue Jays at Red Sox is over. Score: 2-1', 'Phillies win over the Braves, 5-0', 'Phillies 5 Braves 0 final', 'Final: Braves lose to the Phillies in the series opener, 5-0', 'Lightning goaltender pulled, lose to Flyers 4-1', 'Flyers 4 Lightning 1 final', 'Flyers win 4-1']
        cls.embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2'})
        cls.extractor = Extractor(cls.embeddings, 'distilbert-base-cased-distilled-squad')

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        '\n        Cleanup data.\n        '
        if cls.embeddings:
            cls.embeddings.close()

    def testAnswer(self):
        if False:
            i = 10
            return i + 15
        '\n        Test qa extraction with an answer\n        '
        questions = ['What team won the game?', 'What was score?']
        execute = lambda query: self.extractor([(question, query, question, False) for question in questions], self.data)
        answers = execute('Red Sox - Blue Jays')
        self.assertEqual('Blue Jays', answers[0][1])
        self.assertEqual('2-1', answers[1][1])
        question = 'What hockey team won?'
        answers = self.extractor([(question, question, question, False)], self.data)
        self.assertEqual('Flyers', answers[0][1])

    def testEmptyQuery(self):
        if False:
            print('Hello World!')
        '\n        Test an empty extractor queries list\n        '
        self.assertEqual(self.extractor.query(None, None), [])

    def testGeneration(self):
        if False:
            return 10
        '\n        Test support for generator models\n        '
        extractor = Extractor(self.embeddings, 'sshleifer/tiny-gpt2')
        question = 'How many home runs?'
        answers = extractor([(question, question, question, False)], self.data)
        self.assertIsNotNone(answers)

    def testNoAnswer(self):
        if False:
            i = 10
            return i + 15
        '\n        Test qa extraction with no answer\n        '
        question = ''
        answers = self.extractor([(question, question, question, False)], self.data)
        self.assertIsNone(answers[0][1])
        question = 'abcdef'
        answers = self.extractor([(question, question, question, False)], self.data)
        self.assertIsNone(answers[0][1])

    @unittest.skipIf(platform.system() == 'Darwin', 'Quantized models not supported on macOS')
    def testQuantize(self):
        if False:
            while True:
                i = 10
        '\n        Test qa extraction backed by a quantized model\n        '
        extractor = Extractor(self.embeddings, 'distilbert-base-cased-distilled-squad', True)
        question = 'How many home runs?'
        answers = extractor([(question, question, question, True)], self.data)
        self.assertTrue(answers[0][1].startswith('Giants hit 3 HRs'))

    def testOutputs(self):
        if False:
            return 10
        '\n        Test output formatting rules\n        '
        question = 'How many home runs?'
        extractor = Extractor(self.embeddings, 'distilbert-base-cased-distilled-squad', True, output='flatten')
        answers = extractor([(question, question, question, True)], self.data)
        self.assertTrue(answers[0].startswith('Giants hit 3 HRs'))
        extractor = Extractor(self.embeddings, 'distilbert-base-cased-distilled-squad', True, output='reference')
        answers = extractor([(question, question, question, True)], self.data)
        self.assertTrue(self.data[answers[0][2]].startswith('Giants hit 3 HRs'))

    def testSearch(self):
        if False:
            return 10
        '\n        Test qa extraction with an embeddings search for context\n        '
        embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': True})
        embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        extractor = Extractor(embeddings, 'distilbert-base-cased-distilled-squad')
        question = 'How many home runs?'
        answers = extractor([(question, question, question, True)])
        self.assertTrue(answers[0][1].startswith('Giants hit 3 HRs'))

    def testSequences(self):
        if False:
            return 10
        '\n        Test extraction with prompts and a Seq2Seq model\n        '
        extractor = Extractor(self.embeddings, 'google/flan-t5-small')
        prompt = '\n            Answer the following question and return a number.\n            Question: How many HRs?\n            Context:\n        '
        answers = extractor([('prompt', prompt, prompt, False)], self.data)
        self.assertEqual(answers[0][1], '3')

    def testSimilarity(self):
        if False:
            while True:
                i = 10
        '\n        Test qa extraction using a Similarity pipeline to build context\n        '
        extractor = Extractor(Similarity('prajjwal1/bert-medium-mnli'), Questions('distilbert-base-cased-distilled-squad'))
        question = 'How many home runs?'
        answers = extractor([(question, 'HRs', question, True)], self.data)
        self.assertTrue(answers[0][1].startswith('Giants hit 3 HRs'))

    def testSnippet(self):
        if False:
            return 10
        '\n        Test qa extraction with a full answer snippet\n        '
        question = 'How many home runs?'
        answers = self.extractor([(question, question, question, True)], self.data)
        self.assertTrue(answers[0][1].startswith('Giants hit 3 HRs'))

    def testSnippetEmpty(self):
        if False:
            return 10
        '\n        Test snippet method can handle empty parameters\n        '
        self.assertEqual(self.extractor.snippet(None, None), None)

    def testTasks(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test loading models with task parameter\n        '
        for (task, model) in [('language-generation', 'hf-internal-testing/tiny-random-gpt2'), ('sequence-sequence', 'hf-internal-testing/tiny-random-t5')]:
            extractor = Extractor(self.embeddings, model, task=task)
            self.assertIsNotNone(extractor)
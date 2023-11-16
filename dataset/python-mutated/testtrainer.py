"""
Trainer module tests
"""
import os
import unittest
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from txtai.data import Data
from txtai.pipeline import HFTrainer, Labels, Questions, Sequences

class TestTrainer(unittest.TestCase):
    """
    Trainer tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        '\n        Create default datasets.\n        '
        cls.data = [{'text': 'Dogs', 'label': 0}, {'text': 'dog', 'label': 0}, {'text': 'Cats', 'label': 1}, {'text': 'cat', 'label': 1}] * 100

    def testBasic(self):
        if False:
            print('Hello World!')
        '\n        Test training a model with basic parameters\n        '
        trainer = HFTrainer()
        (model, tokenizer) = trainer('google/bert_uncased_L-2_H-128_A-2', self.data)
        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels('cat')[0][0], 1)

    def testCLM(self):
        if False:
            i = 10
            return i + 15
        '\n        Test training a model with causal language modeling.\n        '
        trainer = HFTrainer()
        (model, _) = trainer('hf-internal-testing/tiny-random-gpt2', self.data, maxlength=16, task='language-generation')
        self.assertIsNotNone(model)

    def testCustom(self):
        if False:
            i = 10
            return i + 15
        '\n        Test training a model with custom parameters\n        '
        model = AutoModelForSequenceClassification.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
        tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
        trainer = HFTrainer()
        (model, tokenizer) = trainer((model, tokenizer), self.data, self.data, columns=('text', 'label'), do_eval=True, output_dir=os.path.join(tempfile.gettempdir(), 'trainer'))
        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels('cat')[0][0], 1)

    def testDataFrame(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test training a model with a mock pandas DataFrame\n        '

        class TestDataFrame:
            """
            Test DataFrame
            """

            def __init__(self, data):
                if False:
                    print('Hello World!')
                self.columns = list(data[0].keys())
                self.data = {}
                for column in self.columns:
                    self.data[column] = Values([row[column] for row in data])

            def __getitem__(self, column):
                if False:
                    while True:
                        i = 10
                return self.data[column]

        class Values:
            """
            Test values list
            """

            def __init__(self, values):
                if False:
                    return 10
                self.values = list(values)

            def __getitem__(self, index):
                if False:
                    for i in range(10):
                        print('nop')
                return self.values[index]

            def unique(self):
                if False:
                    while True:
                        i = 10
                '\n                Returns a list of unique values.\n\n                Returns:\n                    unique list of values\n                '
                return set(self.values)
        df = TestDataFrame(self.data)
        trainer = HFTrainer()
        (model, tokenizer) = trainer('google/bert_uncased_L-2_H-128_A-2', df)
        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels('cat')[0][0], 1)

    def testDataset(self):
        if False:
            while True:
                i = 10
        '\n        Test training a model with a mock Hugging Face Dataset\n        '

        class TestDataset(torch.utils.data.Dataset):
            """
            Test Dataset
            """

            def __init__(self, data):
                if False:
                    i = 10
                    return i + 15
                self.data = data
                self.unique = lambda _: [0, 1]

            def __len__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return len(self.data)

            def __getitem__(self, index):
                if False:
                    print('Hello World!')
                return self.data[index]

            def column_names(self):
                if False:
                    for i in range(10):
                        print('nop')
                '\n                Returns column names for this dataset\n\n                Returns:\n                    list of columns\n                '
                return ['text', 'label']

            def map(self, fn, batched, num_proc, remove_columns):
                if False:
                    for i in range(10):
                        print('nop')
                '\n                Map each dataset row using fn.\n\n                Args:\n                    fn: function\n                    batched: batch records\n\n                Returns:\n                    updated Dataset\n                '
                self.data = [fn(x) for x in self.data]
                return self
        ds = TestDataset(self.data)
        trainer = HFTrainer()
        (model, tokenizer) = trainer('google/bert_uncased_L-2_H-128_A-2', ds)
        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels('cat')[0][0], 1)

    def testEmpty(self):
        if False:
            print('Hello World!')
        '\n        Test an empty training data object\n        '
        self.assertIsNone(Data(None, None, None).process(None))

    def testMLM(self):
        if False:
            return 10
        '\n        Test training a model with masked language modeling.\n        '
        trainer = HFTrainer()
        (model, _) = trainer('hf-internal-testing/tiny-random-bert', self.data, task='language-modeling')
        self.assertIsNotNone(model)

    def testMultiLabel(self):
        if False:
            return 10
        '\n        Test training model with labels provided as a list\n        '
        data = []
        for x in self.data:
            data.append({'text': x['text'], 'label': [0.0, 1.0] if x['label'] else [1.0, 0.0]})
        trainer = HFTrainer()
        (model, tokenizer) = trainer('google/bert_uncased_L-2_H-128_A-2', data)
        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels('cat')[0][0], 1)

    def testQA(self):
        if False:
            while True:
                i = 10
        '\n        Test training a QA model\n        '
        data = [{'question': 'What ingredient?', 'context': '1 can whole tomatoes', 'answers': 'tomatoes'}, {'question': 'What ingredient?', 'context': '1 yellow onion', 'answers': 'onion'}, {'question': 'What ingredient?', 'context': '1 red pepper', 'answers': 'pepper'}, {'question': 'What ingredient?', 'context': '1 clove garlic', 'answers': 'garlic'}, {'question': 'What ingredient?', 'context': '1/2 lb beef', 'answers': 'beef'}, {'question': 'What ingredient?', 'context': 'a ' * 500 + '1/2 lb beef', 'answers': 'beef'}, {'question': 'What ingredient?', 'context': 'Forest through the trees', 'answers': None}]
        trainer = HFTrainer()
        (model, tokenizer) = trainer('google/bert_uncased_L-2_H-128_A-2', data, data, task='question-answering', num_train_epochs=10)
        questions = Questions((model, tokenizer), gpu=True)
        self.assertEqual(questions(['What ingredient?'], ['Peel 1 onion'])[0], 'onion')

    def testRegression(self):
        if False:
            while True:
                i = 10
        '\n        Test training a model with a regression (continuous) output\n        '
        data = []
        for x in self.data:
            data.append({'text': x['text'], 'label': x['label'] + 0.1})
        trainer = HFTrainer()
        (model, tokenizer) = trainer('google/bert_uncased_L-2_H-128_A-2', data)
        labels = Labels((model, tokenizer), dynamic=False)
        self.assertGreater(labels('cat')[0][1], 0.5)

    def testRTD(self):
        if False:
            i = 10
            return i + 15
        '\n        Test training a language model with replaced token detection\n        '
        output = os.path.join(tempfile.gettempdir(), 'trainer.rtd')
        trainer = HFTrainer()
        (model, _) = trainer('hf-internal-testing/tiny-random-electra', self.data, task='token-detection', output_dir=output)
        self.assertIsNotNone(model)
        self.assertTrue(os.path.exists(os.path.join(output, 'generator')))
        self.assertTrue(os.path.exists(os.path.join(output, 'discriminator')))

    def testSeqSeq(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test training a sequence-sequence model\n        '
        data = [{'source': 'Running again', 'target': 'Sleeping again'}, {'source': 'Run', 'target': 'Sleep'}, {'source': 'running', 'target': 'sleeping'}]
        trainer = HFTrainer()
        (model, tokenizer) = trainer('t5-small', data, task='sequence-sequence', prefix='translate Run to Sleep: ', learning_rate=0.001)
        sequences = Sequences((model, tokenizer))
        result = sequences('translate Run to Sleep: run')
        self.assertEqual(result.lower(), 'sleep')
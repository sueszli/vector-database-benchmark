"""
ONNX module tests
"""
import os
import tempfile
import unittest
from unittest.mock import patch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from txtai.embeddings import Embeddings
from txtai.pipeline import HFOnnx, HFTrainer, Labels, MLOnnx, Questions

class TestOnnx(unittest.TestCase):
    """
    ONNX tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create default datasets.\n        '
        cls.data = [{'text': 'Dogs', 'label': 0}, {'text': 'dog', 'label': 0}, {'text': 'Cats', 'label': 1}, {'text': 'cat', 'label': 1}] * 100

    def testDefault(self):
        if False:
            while True:
                i = 10
        '\n        Test exporting an ONNX model with default parameters\n        '
        onnx = HFOnnx()
        model = onnx('google/bert_uncased_L-2_H-128_A-2')
        self.assertGreater(len(model), 0)

    def testClassification(self):
        if False:
            while True:
                i = 10
        '\n        Test exporting a classification model to ONNX and running inference\n        '
        path = 'google/bert_uncased_L-2_H-128_A-2'
        trainer = HFTrainer()
        (model, tokenizer) = trainer(path, self.data)
        output = os.path.join(tempfile.gettempdir(), 'onnx')
        onnx = HFOnnx()
        model = onnx((model, tokenizer), 'text-classification', output, True)
        labels = Labels((model, path), dynamic=False)
        self.assertEqual(labels('cat')[0][0], 1)

    @patch('onnxruntime.get_available_providers')
    @patch('torch.cuda.is_available')
    def testPooling(self, cuda, providers):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test exporting a pooling model to ONNX and running inference\n        '
        path = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
        onnx = HFOnnx()
        model = onnx(path, 'pooling', quantize=True)
        cuda.return_value = False
        providers.return_value = ['CPUExecutionProvider']
        embeddings = Embeddings({'path': model, 'tokenizer': path})
        self.assertEqual(embeddings.similarity('animal', ['dog', 'book', 'rug'])[0][0], 0)
        cuda.return_value = False
        providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        embeddings = Embeddings({'path': model, 'tokenizer': path})
        self.assertIsNotNone(embeddings)
        cuda.return_value = True
        providers.return_value = ['CPUExecutionProvider']
        embeddings = Embeddings({'path': model, 'tokenizer': path})
        self.assertIsNotNone(embeddings)
        cuda.return_value = True
        providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        embeddings = Embeddings({'path': model, 'tokenizer': path})
        self.assertIsNotNone(embeddings)

    def testQA(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test exporting a QA model to ONNX and running inference\n        '
        path = 'distilbert-base-cased-distilled-squad'
        onnx = HFOnnx()
        model = onnx(path, 'question-answering')
        questions = Questions((model, path))
        self.assertEqual(questions(['What is the price?'], ['The price is $30'])[0], '$30')

    def testScikit(self):
        if False:
            i = 10
            return i + 15
        '\n        Test exporting a scikit-learn model to ONNX and running inference\n        '

        def tokenizer(inputs, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(inputs, str):
                inputs = [inputs]
            return {'input_ids': [[x] for x in inputs]}
        model = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression())])
        model.fit([x['text'] for x in self.data], [x['label'] for x in self.data])
        onnx = MLOnnx()
        model = onnx(model)
        labels = Labels((model, tokenizer), dynamic=False)
        self.assertEqual(labels('cat')[0][0], 1)

    @unittest.skipIf(os.name == 'nt', 'testZeroShot skipped on Windows')
    def testZeroShot(self):
        if False:
            i = 10
            return i + 15
        '\n        Test exporting a zero shot classification model to ONNX and running inference\n        '
        path = 'prajjwal1/bert-medium-mnli'
        onnx = HFOnnx()
        model = onnx(path, 'zero-shot-classification', quantize=True)
        labels = Labels((model, path))
        self.assertEqual(labels('That is great news', ['negative', 'positive'])[0][0], 1)
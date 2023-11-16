"""
Pipeline API module tests
"""
import os
import tempfile
import unittest
import urllib
from unittest.mock import patch
from fastapi.testclient import TestClient
from txtai.api import API, app, start
from utils import Utils
PIPELINES = '\n# Image captions\ncaption:\n\n# Entity extraction\nentity:\n    path: dslim/bert-base-NER\n\n# Label settings\nlabels:\n    path: prajjwal1/bert-medium-mnli\n\n# Image objects\nobjects:\n\n# Text segmentation\nsegmentation:\n    sentences: true\n\n# Enable pipeline similarity backed by zero shot classifier\nsimilarity:\n\n# Summarization\nsummary:\n    path: t5-small\n\n# Tabular\ntabular:\n\n# Text extraction\ntextractor:\n\n# Transcription\ntranscription:\n\n# Translation:\ntranslation:\n'

@unittest.skipIf(os.name == 'nt', 'TestPipelines skipped on Windows')
class TestPipelines(unittest.TestCase):
    """
    API tests for pipelines.
    """

    @staticmethod
    @patch.dict(os.environ, {'CONFIG': os.path.join(tempfile.gettempdir(), 'testapi.yml'), 'API_CLASS': 'txtai.api.API'})
    def start():
        if False:
            while True:
                i = 10
        '\n        Starts a mock FastAPI client.\n        '
        config = os.path.join(tempfile.gettempdir(), 'testapi.yml')
        with open(config, 'w', encoding='utf-8') as output:
            output.write(PIPELINES)
        client = TestClient(app)
        start()
        return client

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        '\n        Create API client on creation of class.\n        '
        cls.client = TestPipelines.start()
        cls.data = ['US tops 5 million confirmed virus cases', "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", 'Beijing mobilises invasion craft along coast as Taiwan tensions escalate', 'The National Park Service warns against sacrificing slower friends in a bear attack', 'Maine man wins $1M from $25 lottery ticket', 'Make huge profits without work, earn up to $100,000 a day']
        cls.text = "Search is the base of many applications. Once data starts to pile up, users want to be able to find it. It's the foundation of the internet and an ever-growing challenge that is never solved or done. The field of Natural Language Processing (NLP) is rapidly evolving with a number of new developments. Large-scale general language models are an exciting new capability allowing us to add amazing functionality quickly with limited compute and people. Innovation continues with new models and advancements coming in at what seems a weekly basis. This article introduces txtai, an AI-powered search engine that enables Natural Language Understanding (NLU) based search in any application."

    def testCaption(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test caption via API\n        '
        caption = self.client.get(f'caption?file={Utils.PATH}/books.jpg').json()
        self.assertEqual(caption, 'a book shelf filled with books and a stack of books')

    def testCaptionBatch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test batch caption via API\n        '
        path = Utils.PATH + '/books.jpg'
        captions = self.client.post('batchcaption', json=[path, path]).json()
        self.assertEqual(captions, ['a book shelf filled with books and a stack of books'] * 2)

    def testEntity(self):
        if False:
            print('Hello World!')
        '\n        Test entity extraction via API\n        '
        entities = self.client.get(f'entity?text={self.data[1]}').json()
        self.assertEqual([e[0] for e in entities], ['Canada', 'Manhattan'])

    def testEntityBatch(self):
        if False:
            return 10
        '\n        Test batch entity via API\n        '
        entities = self.client.post('batchentity', json=[self.data[1]]).json()
        self.assertEqual([e[0] for e in entities[0]], ['Canada', 'Manhattan'])

    def testEmpty(self):
        if False:
            i = 10
            return i + 15
        '\n        Test empty API configuration\n        '
        api = API({})
        self.assertIsNone(api.label('test', ['test']))
        self.assertIsNone(api.pipeline('junk', 'test'))

    def testLabel(self):
        if False:
            while True:
                i = 10
        '\n        Test label via API\n        '
        labels = self.client.post('label', json={'text': 'this is the best sentence ever', 'labels': ['positive', 'negative']}).json()
        self.assertEqual(labels[0]['id'], 0)

    def testLabelBatch(self):
        if False:
            return 10
        '\n        Test batch label via API\n        '
        labels = self.client.post('batchlabel', json={'texts': ['this is the best sentence ever', 'This is terrible'], 'labels': ['positive', 'negative']}).json()
        results = [l[0]['id'] for l in labels]
        self.assertEqual(results, [0, 1])

    def testObjects(self):
        if False:
            while True:
                i = 10
        '\n        Test objects via API\n        '
        objects = self.client.get(f'objects?file={Utils.PATH}/books.jpg').json()
        self.assertEqual(objects[0][0], 'book')

    def testObjectsBatch(self):
        if False:
            while True:
                i = 10
        '\n        Test batch objects via API\n        '
        path = Utils.PATH + '/books.jpg'
        objects = self.client.post('batchobjects', json=[path, path]).json()
        self.assertEqual([o[0][0] for o in objects], ['book'] * 2)

    def testSegment(self):
        if False:
            print('Hello World!')
        '\n        Test segmentation via API\n        '
        text = self.client.get('segment?text=This is a test. And another test.').json()
        self.assertEqual(len(text), 2)

    def testSegmentBatch(self):
        if False:
            print('Hello World!')
        '\n        Test batch segmentation via API\n        '
        text = 'This is a test. And another test.'
        texts = self.client.post('batchsegment', json=[text, text]).json()
        self.assertEqual(len(texts), 2)
        self.assertEqual(len(texts[0]), 2)

    def testSimilarity(self):
        if False:
            return 10
        '\n        Test similarity via API\n        '
        uid = self.client.post('similarity', json={'query': 'feel good story', 'texts': self.data}).json()[0]['id']
        self.assertEqual(self.data[uid], self.data[4])

    def testSimilarityBatch(self):
        if False:
            print('Hello World!')
        '\n        Test batch similarity via API\n        '
        results = self.client.post('batchsimilarity', json={'queries': ['feel good story', 'climate change'], 'texts': self.data}).json()
        uids = [result[0]['id'] for result in results]
        self.assertEqual(uids, [4, 1])

    def testSummary(self):
        if False:
            print('Hello World!')
        '\n        Test summary via API\n        '
        summary = self.client.get(f'summary?text={urllib.parse.quote(self.text)}&minlength=15&maxlength=15').json()
        self.assertEqual(summary, 'the field of natural language processing (NLP) is rapidly evolving')

    def testSummaryBatch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test batch summary via API\n        '
        summaries = self.client.post('batchsummary', json={'texts': [self.text, self.text], 'minlength': 15, 'maxlength': 15}).json()
        self.assertEqual(summaries, ['the field of natural language processing (NLP) is rapidly evolving'] * 2)

    def testTabular(self):
        if False:
            print('Hello World!')
        '\n        Test tabular via API\n        '
        results = self.client.get(f'tabular?file={Utils.PATH}/tabular.csv').json()
        self.assertEqual(len(results), 6)

    def testTabularBatch(self):
        if False:
            while True:
                i = 10
        '\n        Test batch tabular via API\n        '
        path = Utils.PATH + '/tabular.csv'
        results = self.client.post('batchtabular', json=[path, path]).json()
        self.assertEqual((len(results[0]), len(results[1])), (6, 6))

    def testTextractor(self):
        if False:
            print('Hello World!')
        '\n        Test textractor via API\n        '
        text = self.client.get(f'textract?file={Utils.PATH}/article.pdf').json()
        self.assertEqual(len(text), 2301)

    def testTextractorBatch(self):
        if False:
            return 10
        '\n        Test batch textractor via API\n        '
        path = Utils.PATH + '/article.pdf'
        texts = self.client.post('batchtextract', json=[path, path]).json()
        self.assertEqual((len(texts[0]), len(texts[1])), (2301, 2301))

    def testTranscribe(self):
        if False:
            while True:
                i = 10
        '\n        Test transcribe via API\n        '
        text = self.client.get(f'transcribe?file={Utils.PATH}/Make_huge_profits.wav').json()
        self.assertEqual(text, 'Make huge profits without working make up to one hundred thousand dollars a day')

    def testTranscribeBatch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test batch transcribe via API\n        '
        path = Utils.PATH + '/Make_huge_profits.wav'
        texts = self.client.post('batchtranscribe', json=[path, path]).json()
        self.assertEqual(texts, ['Make huge profits without working make up to one hundred thousand dollars a day'] * 2)

    def testTranslate(self):
        if False:
            return 10
        '\n        Test translate via API\n        '
        translation = self.client.get(f"translate?text={urllib.parse.quote('This is a test translation into Spanish')}&target=es").json()
        self.assertEqual(translation, 'Esta es una traducción de prueba al español')

    def testTranslateBatch(self):
        if False:
            return 10
        '\n        Test batch translate via API\n        '
        text = 'This is a test translation into Spanish'
        translations = self.client.post('batchtranslate', json={'texts': [text, text], 'target': 'es'}).json()
        self.assertEqual(translations, ['Esta es una traducción de prueba al español'] * 2)
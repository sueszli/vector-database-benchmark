"""
Embeddings API module tests
"""
import os
import tempfile
import unittest
import urllib.parse
from unittest.mock import patch
from fastapi.testclient import TestClient
from txtai.api import API, app, start
INDEX = '\n# Index file path\npath: %s\n\n# Allow indexing of documents\nwritable: True\n\n# Questions settings\nquestions:\n    path: distilbert-base-cased-distilled-squad\n\n# Embeddings settings\nembeddings:\n    path: sentence-transformers/nli-mpnet-base-v2\n\n# Extractor settings\nextractor:\n    path: questions\n'
READONLY = '\n# Index file path\npath: %s\n\n# Allow indexing of documents\nwritable: False\n'
FUNCTIONS = '\npathignore: %s\n\nwritable: True\n\n# Embeddings settings\nembeddings:\n    path: sentence-transformers/nli-mpnet-base-v2\n    content: True\n    functions:\n        - testapi.testembeddings.Elements\n        - name: length\n          argcount: 1\n          function: testapi.testembeddings.length\n        - name: ann\n          function: ann\n    transform: testapi.testembeddings.transform\n'

class TestEmbeddings(unittest.TestCase):
    """
    API tests for embeddings indices.
    """

    @staticmethod
    @patch.dict(os.environ, {'CONFIG': os.path.join(tempfile.gettempdir(), 'testapi.yml'), 'API_CLASS': 'txtai.api.API'})
    def start(yaml):
        if False:
            for i in range(10):
                print('nop')
        '\n        Starts a mock FastAPI client.\n\n        Args:\n            yaml: input configuration\n        '
        config = os.path.join(tempfile.gettempdir(), 'testapi.yml')
        index = os.path.join(tempfile.gettempdir(), 'testapi')
        with open(config, 'w', encoding='utf-8') as output:
            output.write(yaml % index)
        client = TestClient(app)
        start()
        return client

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        '\n        Create API client on creation of class.\n        '
        cls.client = TestEmbeddings.start(INDEX)
        cls.data = ['US tops 5 million confirmed virus cases', "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", 'Beijing mobilises invasion craft along coast as Taiwan tensions escalate', 'The National Park Service warns against sacrificing slower friends in a bear attack', 'Maine man wins $1M from $25 lottery ticket', 'Make huge profits without work, earn up to $100,000 a day']
        cls.client.post('add', json=[{'id': x, 'text': row} for (x, row) in enumerate(cls.data)])
        cls.client.get('index')

    def testCount(self):
        if False:
            print('Hello World!')
        '\n        Test count via API\n        '
        self.assertEqual(self.client.get('count').json(), 6)

    def testDelete(self):
        if False:
            while True:
                i = 10
        '\n        Test delete via API\n        '
        ids = self.client.post('delete', json=[4]).json()
        self.assertEqual(ids, [4])
        query = urllib.parse.quote('feel good story')
        uid = self.client.get(f'search?query={query}&limit=1').json()[0]['id']
        self.assertEqual(self.client.get('count').json(), 5)
        self.assertEqual(uid, 5)
        self.client.post('add', json=[{'id': x, 'text': row} for (x, row) in enumerate(self.data)])
        self.client.get('index')

    def testEmpty(self):
        if False:
            i = 10
            return i + 15
        '\n        Test empty API configuration\n        '
        api = API({'writable': True})
        self.assertIsNone(api.search('test', None))
        self.assertIsNone(api.batchsearch(['test'], None))
        self.assertIsNone(api.delete(['test']))
        self.assertIsNone(api.count())
        self.assertIsNone(api.similarity('test', ['test']))
        self.assertIsNone(api.batchsimilarity(['test'], ['test']))
        self.assertIsNone(api.explain('test'))
        self.assertIsNone(api.batchexplain(['test']))
        self.assertIsNone(api.transform('test'))
        self.assertIsNone(api.batchtransform(['test']))
        self.assertIsNone(api.extract(['test'], ['test']))

    def testExtractor(self):
        if False:
            print('Hello World!')
        '\n        Test qa extraction via API\n        '
        data = ['Giants hit 3 HRs to down Dodgers', 'Giants 5 Dodgers 4 final', 'Dodgers drop Game 2 against the Giants, 5-4', 'Blue Jays beat Red Sox final score 2-1', 'Red Sox lost to the Blue Jays, 2-1', 'Blue Jays at Red Sox is over. Score: 2-1', 'Phillies win over the Braves, 5-0', 'Phillies 5 Braves 0 final', 'Final: Braves lose to the Phillies in the series opener, 5-0', 'Lightning goaltender pulled, lose to Flyers 4-1', 'Flyers 4 Lightning 1 final', 'Flyers win 4-1']
        questions = ['What team won the game?', 'What was score?']
        execute = lambda query: self.client.post('extract', json={'queue': [{'name': question, 'query': query, 'question': question, 'snippet': False} for question in questions], 'texts': data}).json()
        answers = execute('Red Sox - Blue Jays')
        self.assertEqual('Blue Jays', answers[0]['answer'])
        self.assertEqual('2-1', answers[1]['answer'])
        question = 'What hockey team won?'
        answers = self.client.post('extract', json={'queue': [{'name': question, 'query': question, 'question': question, 'snippet': False}], 'texts': data}).json()
        self.assertEqual('Flyers', answers[0]['answer'])

    def testReindex(self):
        if False:
            return 10
        '\n        Test reindex via API\n        '
        self.client.post('reindex', json={'config': {'path': 'sentence-transformers/nli-mpnet-base-v2'}})
        query = urllib.parse.quote('feel good story')
        uid = self.client.get(f'search?query={query}&limit=1').json()[0]['id']
        self.assertEqual(uid, 4)
        self.client.post('add', json=[{'id': x, 'text': row} for (x, row) in enumerate(self.data)])
        self.client.get('index')

    def testSearch(self):
        if False:
            print('Hello World!')
        '\n        Test search via API\n        '
        query = urllib.parse.quote('feel good story')
        uid = self.client.get(f'search?query={query}&limit=1').json()[0]['id']
        self.assertEqual(uid, 4)

    def testSearchBatch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test batch search via API\n        '
        results = self.client.post('batchsearch', json={'queries': ['feel good story', 'climate change'], 'limit': 1}).json()
        uids = [result[0]['id'] for result in results]
        self.assertEqual(uids, [4, 1])

    def testSimilarity(self):
        if False:
            return 10
        '\n        Test similarity via API\n        '
        uid = self.client.post('similarity', json={'query': 'feel good story', 'texts': self.data}).json()[0]['id']
        self.assertEqual(uid, 4)

    def testSimilarityBatch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test batch similarity via API\n        '
        results = self.client.post('batchsimilarity', json={'queries': ['feel good story', 'climate change'], 'texts': self.data}).json()
        uids = [result[0]['id'] for result in results]
        self.assertEqual(uids, [4, 1])

    def testTransform(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test embeddings transform via API\n        '
        self.assertEqual(len(self.client.get('transform?text=testembed').json()), 768)

    def testTransformBatch(self):
        if False:
            i = 10
            return i + 15
        '\n        Test batch embeddings transform via API\n        '
        embeddings = self.client.post('batchtransform', json=self.data).json()
        self.assertEqual(len(embeddings), len(self.data))
        self.assertEqual(len(embeddings[0]), 768)

    def testUpsert(self):
        if False:
            while True:
                i = 10
        '\n        Test upsert via API\n        '
        self.client.post('add', json=[{'id': 0, 'text': 'Feel good story: baby panda born'}])
        self.client.get('upsert')
        query = urllib.parse.quote('feel good story')
        uid = self.client.get(f'search?query={query}&limit=1').json()[0]['id']
        self.assertEqual(uid, 0)
        self.client.post('add', json=[{'id': x, 'text': row} for (x, row) in enumerate(self.data)])
        self.client.get('index')

    def testViewOnly(self):
        if False:
            i = 10
            return i + 15
        '\n        Test read-only API instance\n        '
        self.client = TestEmbeddings.start(READONLY)
        query = urllib.parse.quote('feel good story')
        uid = self.client.get(f'search?query={query}&limit=1').json()[0]['id']
        self.assertEqual(uid, 4)
        uid = self.client.post('similarity', json={'query': 'feel good story', 'texts': self.data}).json()[0]['id']
        self.assertEqual(uid, 4)
        self.assertEqual(self.client.post('add', json=[{'id': 0, 'text': 'test'}]).status_code, 403)
        self.assertEqual(self.client.get('index').status_code, 403)
        self.assertEqual(self.client.get('upsert').status_code, 403)
        self.assertEqual(self.client.post('delete', json=[0]).status_code, 403)
        self.assertEqual(self.client.post('reindex', json={'config': {'path': 'sentence-transformers/nli-mpnet-base-v2'}}).status_code, 403)

    def testXFunctions(self):
        if False:
            print('Hello World!')
        '\n        Test API instance with custom functions\n        '
        self.client = TestEmbeddings.start(FUNCTIONS)
        self.client.post('add', json=[{'id': x, 'text': row} for (x, row) in enumerate(self.data)])
        self.client.get('index')
        query = urllib.parse.quote("select elements('text') length from txtai limit 1")
        self.assertEqual(self.client.get(f'search?query={query}').json()[0]['length'], 4)
        query = urllib.parse.quote("select length('text') length from txtai limit 1")
        self.assertEqual(self.client.get(f'search?query={query}').json()[0]['length'], 4)

    def testXPlain(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test API instance with explain methods\n        '
        results = self.client.post('explain', json={'query': 'feel good story', 'limit': 1}).json()
        self.assertEqual(results[0]['text'], self.data[4])
        self.assertIsNotNone(results[0].get('tokens'))

    def testXPlainBatch(self):
        if False:
            print('Hello World!')
        '\n        Test batch query explain via API\n        '
        results = self.client.post('batchexplain', json={'queries': ['feel good story', 'climate change'], 'limit': 1}).json()
        text = [result[0]['text'] for result in results]
        self.assertEqual(text, [self.data[4], self.data[1]])
        self.assertIsNotNone(results[0][0].get('tokens'))

class Elements:
    """
    Custom SQL function as callable object.
    """

    def __call__(self, text):
        if False:
            return 10
        return length(text)

def transform(document):
    if False:
        return 10
    '\n    Custom transform function.\n    '
    return document

def length(text):
    if False:
        while True:
            i = 10
    '\n    Custom SQL function.\n    '
    return len(text)
"""
Cloud module tests
"""
import os
import tempfile
import time
import unittest
from unittest.mock import patch
from txtai.cloud import Cloud
from txtai.embeddings import Embeddings

class TestCloud(unittest.TestCase):
    """
    Cloud tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize test data.\n        '
        cls.data = ['US tops 5 million confirmed virus cases', "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", 'Beijing mobilises invasion craft along coast as Taiwan tensions escalate', 'The National Park Service warns against sacrificing slower friends in a bear attack', 'Maine man wins $1M from $25 lottery ticket', 'Make huge profits without work, earn up to $100,000 a day']
        cls.embeddings = Embeddings({'format': 'json', 'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': True})

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        '\n        Cleanup data.\n        '
        if cls.embeddings:
            cls.embeddings.close()

    def testCustom(self):
        if False:
            i = 10
            return i + 15
        '\n        Test custom provider\n        '
        self.runHub('txtai.cloud.HuggingFaceHub')

    def testHub(self):
        if False:
            print('Hello World!')
        '\n        Test huggingface-hub integration\n        '
        self.runHub('huggingface-hub')

    def testInvalidProvider(self):
        if False:
            i = 10
            return i + 15
        '\n        Test invalid provider identifier\n        '
        with self.assertRaises(ImportError):
            embeddings = Embeddings()
            embeddings.load(provider='ProviderNoExist', container='Invalid')

    def testNotImplemented(self):
        if False:
            i = 10
            return i + 15
        '\n        Test exceptions for non-implemented methods\n        '
        cloud = Cloud({})
        self.assertRaises(NotImplementedError, cloud.exists, None)
        self.assertRaises(NotImplementedError, cloud.load, None)
        self.assertRaises(NotImplementedError, cloud.save, None)

    def testObjectStorage(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test object storage integration\n        '
        for path in ['cloud.object', 'cloud.object.tar.gz']:
            self.runTests(path, {'provider': 'local', 'container': f'cloud.{time.time()}', 'key': tempfile.gettempdir()})

    @patch('huggingface_hub.hf_hub_download')
    @patch('huggingface_hub.get_hf_file_metadata')
    @patch('huggingface_hub.upload_file')
    @patch('huggingface_hub.create_repo')
    def runHub(self, provider, create, upload, metadata, download):
        if False:
            for i in range(10):
                print('nop')
        "\n        Run huggingface-hub tests. This method mocks write operations since a token won't be available.\n        "

        def filemeta(url, token):
            if False:
                i = 10
                return i + 15
            return (url, token) if 'Invalid' not in url else None

        def filedownload(**kwargs):
            if False:
                print('Hello World!')
            if 'Invalid' in kwargs['repo_id']:
                raise FileNotFoundError
            return attributes if kwargs['filename'] == '.gitattributes' else index
        create.return_value = None
        upload.return_value = None
        metadata.side_effect = filemeta
        download.side_effect = filedownload
        self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        index = os.path.join(tempfile.gettempdir(), f'cloud.{provider}.tar.gz')
        self.embeddings.save(index)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write('*.bin filter=lfs diff=lfs merge=lfs -text\n')
            attributes = tmp.name
        for path in [f'cloud.{provider}', f'cloud.{provider}.tar.gz']:
            self.runTests(path, {'provider': provider, 'container': 'neuml/txtai-intro'})

    def runTests(self, path, cloud):
        if False:
            i = 10
            return i + 15
        '\n        Runs a series of cloud sync tests.\n        '
        self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
        index = os.path.join(tempfile.gettempdir(), path)
        invalid = cloud.copy()
        invalid['container'] = 'InvalidPathToTest'
        self.assertFalse(self.embeddings.exists(index, invalid))
        with self.assertRaises(Exception):
            self.embeddings.load(index, invalid)
        self.embeddings.save(index, cloud)
        self.assertTrue(self.embeddings.exists(index, cloud))
        self.assertTrue(self.embeddings.exists(index))
        self.embeddings.load(index, cloud)
        result = self.embeddings.search('feel good story', 1)[0]
        self.assertEqual(result['text'], self.data[4])
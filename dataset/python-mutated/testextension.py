"""
Extension module tests
"""
import os
import tempfile
import unittest
from unittest.mock import patch
from fastapi import APIRouter
from fastapi.testclient import TestClient
from txtai.api import application, Extension
from txtai.pipeline import Pipeline
PIPELINES = '\ntestapi.testextension.SamplePipeline:\n'

class SampleRouter:
    """
    Sample API router.
    """
    router = APIRouter()

    @staticmethod
    @router.get('/sample')
    def sample(text: str):
        if False:
            return 10
        '\n        Calls sample pipeline.\n\n        Args:\n            text: input text\n\n        Returns:\n            formatted text\n        '
        return application.get().pipeline('testapi.testextension.SamplePipeline', (text,))

class SampleExtension(Extension):
    """
    Sample API extension.
    """

    def __call__(self, app):
        if False:
            for i in range(10):
                print('nop')
        app.include_router(SampleRouter().router)

class SamplePipeline(Pipeline):
    """
    Sample pipeline.
    """

    def __call__(self, text):
        if False:
            i = 10
            return i + 15
        return text.lower()

class TestExtension(unittest.TestCase):
    """
    API tests for extensions.
    """

    @staticmethod
    @patch.dict(os.environ, {'CONFIG': os.path.join(tempfile.gettempdir(), 'testapi.yml'), 'API_CLASS': 'txtai.api.API', 'EXTENSIONS': 'testapi.testextension.SampleExtension'})
    def start():
        if False:
            i = 10
            return i + 15
        '\n        Starts a mock FastAPI client.\n        '
        config = os.path.join(tempfile.gettempdir(), 'testapi.yml')
        with open(config, 'w', encoding='utf-8') as output:
            output.write(PIPELINES)
        client = TestClient(application.app)
        application.start()
        return client

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create API client on creation of class.\n        '
        cls.client = TestExtension.start()

    def testEmpty(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test an empty extension\n        '
        extension = Extension()
        self.assertIsNone(extension(None))

    def testExtension(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a pipeline extension\n        '
        text = self.client.get('sample?text=Test%20String').json()
        self.assertEqual(text, 'test string')
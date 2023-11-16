"""
Optional module tests
"""
import sys
import unittest
from transformers import Trainer

class TestOptional(unittest.TestCase):
    """
    Optional tests. Simulates optional dependencies not being installed.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        '\n        Simulate optional packages not being installed\n        '
        modules = ['annoy', 'croniter', 'duckdb', 'fastapi', 'fasttext', 'hnswlib', 'imagehash', 'nltk', 'libcloud.storage.providers', 'networkx', 'onnxmltools', 'onnxruntime', 'onnxruntime.quantization', 'pandas', 'PIL', 'rich', 'sklearn.decomposition', 'sentence_transformers', 'soundfile', 'sqlalchemy', 'tika', 'ttstokenizer', 'xmltodict']
        modules = modules + [key for key in sys.modules if key.startswith('txtai')]
        cls.modules = {module: None for module in modules}
        for module in cls.modules:
            if module in sys.modules:
                cls.modules[module] = sys.modules[module]
            if 'txtai' in module:
                if module in sys.modules:
                    del sys.modules[module]
            else:
                sys.modules[module] = None

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        '\n        Resets modules environment back to initial state.\n        '
        for (key, value) in cls.modules.items():
            if value:
                sys.modules[key] = value
            else:
                del sys.modules[key]

    def testApi(self):
        if False:
            return 10
        '\n        Test missing api dependencies\n        '
        with self.assertRaises(ImportError):
            import txtai.api

    def testConsole(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test missing console dependencies\n        '
        from txtai.console import Console
        with self.assertRaises(ImportError):
            Console()

    def testCloud(self):
        if False:
            i = 10
            return i + 15
        '\n        Test missing cloud dependencies\n        '
        from txtai.cloud import ObjectStorage
        with self.assertRaises(ImportError):
            ObjectStorage(None)

    def testDatabase(self):
        if False:
            print('Hello World!')
        '\n        Test missing database dependencies\n        '
        from txtai.database import Client, DuckDB, ImageEncoder
        with self.assertRaises(ImportError):
            Client({})
        with self.assertRaises(ImportError):
            DuckDB({})
        with self.assertRaises(ImportError):
            ImageEncoder()

    def testGraph(self):
        if False:
            while True:
                i = 10
        '\n        Test missing graph dependencies\n        '
        from txtai.graph import GraphFactory
        with self.assertRaises(ImportError):
            GraphFactory.create({'backend': 'networkx'})

    def testModel(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test missing model dependencies\n        '
        from txtai.embeddings import Reducer
        from txtai.models import OnnxModel
        with self.assertRaises(ImportError):
            Reducer()
        with self.assertRaises(ImportError):
            OnnxModel(None)

    def testPipeline(self):
        if False:
            while True:
                i = 10
        '\n        Test missing pipeline dependencies\n        '
        from txtai.pipeline import Caption, HFOnnx, ImageHash, MLOnnx, Objects, Segmentation, Tabular, Textractor, TextToSpeech, Transcription, Translation
        with self.assertRaises(ImportError):
            Caption()
        with self.assertRaises(ImportError):
            HFOnnx()('google/bert_uncased_L-2_H-128_A-2', quantize=True)
        with self.assertRaises(ImportError):
            ImageHash()
        with self.assertRaises(ImportError):
            MLOnnx()
        with self.assertRaises(ImportError):
            Objects()
        with self.assertRaises(ImportError):
            Segmentation()
        with self.assertRaises(ImportError):
            Tabular()
        with self.assertRaises(ImportError):
            Textractor()
        with self.assertRaises(ImportError):
            TextToSpeech()
        with self.assertRaises(ImportError):
            Transcription()
        with self.assertRaises(ImportError):
            Translation().detect(['test'])

    def testSimilarity(self):
        if False:
            return 10
        '\n        Test missing similarity dependencies\n        '
        from txtai.ann import ANNFactory
        from txtai.vectors import VectorsFactory
        with self.assertRaises(ImportError):
            ANNFactory.create({'backend': 'annoy'})
        with self.assertRaises(ImportError):
            ANNFactory.create({'backend': 'hnsw'})
        with self.assertRaises(ImportError):
            VectorsFactory.create({'method': 'words'}, None)
        with self.assertRaises(ImportError):
            VectorsFactory.create({'method': 'sentence-transformers', 'path': ''}, None)

    def testWorkflow(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test missing workflow dependencies\n        '
        from txtai.workflow import ExportTask, ImageTask, ServiceTask, StorageTask, Workflow
        with self.assertRaises(ImportError):
            ExportTask()
        with self.assertRaises(ImportError):
            ImageTask()
        with self.assertRaises(ImportError):
            ServiceTask()
        with self.assertRaises(ImportError):
            StorageTask()
        with self.assertRaises(ImportError):
            Workflow([], workers=1).schedule(None, [])
"""
Workflow module tests
"""
import contextlib
import glob
import io
import os
import tempfile
import sys
import unittest
import numpy as np
import torch
from txtai.api import API
from txtai.embeddings import Documents, Embeddings
from txtai.pipeline import Nop, Segmentation, Summary, Translation, Textractor
from txtai.workflow import Workflow, Task, ConsoleTask, ExportTask, ExtractorTask, FileTask, ImageTask, RetrieveTask, StorageTask, TemplateTask, WorkflowTask
from utils import Utils

class TestWorkflow(unittest.TestCase):
    """
    Workflow tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        '\n        Initialize test data.\n        '
        cls.config = '\n        # Embeddings index\n        writable: true\n        embeddings:\n            scoring: bm25\n            path: google/bert_uncased_L-2_H-128_A-2\n            content: true\n\n        # Text segmentation\n        segmentation:\n            sentences: true\n\n        # Workflow definitions\n        workflow:\n            index:\n                tasks:\n                    - action: segmentation\n                    - action: index\n            search:\n                tasks:\n                    - search\n            transform:\n                tasks:\n                    - transform\n        '

    @unittest.skipIf(os.name == 'nt', 'testBaseWorkflow skipped on Windows')
    def testBaseWorkflow(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a basic workflow\n        '
        translate = Translation()
        workflow = Workflow([Task(lambda x: translate(x, 'es'))])
        results = list(workflow(['The sky is blue', 'Forest through the trees']))
        self.assertEqual(len(results), 2)

    def testChainWorkflow(self):
        if False:
            print('Hello World!')
        '\n        Test a chain of workflows\n        '
        workflow1 = Workflow([Task(lambda x: [y * 2 for y in x])])
        workflow2 = Workflow([Task(lambda x: [y - 1 for y in x])], batch=4)
        results = list(workflow2(workflow1([1, 2, 4, 8, 16, 32])))
        self.assertEqual(results, [1, 3, 7, 15, 31, 63])

    @unittest.skipIf(os.name == 'nt', 'testComplexWorkflow skipped on Windows')
    def testComplexWorkflow(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a complex workflow\n        '
        textractor = Textractor(paragraphs=True, minlength=150, join=True)
        summary = Summary('t5-small')
        embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2'})
        documents = Documents()

        def index(x):
            if False:
                return 10
            documents.add(x)
            return x
        articles = Workflow([FileTask(textractor), Task(lambda x: summary(x, maxlength=15))])
        tasks = [WorkflowTask(articles, '.\\.pdf$'), Task(index, unpack=False)]
        data = ['file://' + Utils.PATH + '/article.pdf', 'Workflows can process audio files, documents and snippets']
        data = [(x, element, None) for (x, element) in enumerate(data)]
        workflow = Workflow(tasks)
        data = list(workflow(data))
        embeddings.index(documents)
        documents.close()
        (index, _) = embeddings.search('search text', 1)[0]
        self.assertEqual(index, 0)
        self.assertEqual(data[0][1], 'txtai builds an AI-powered index over sections')

    @unittest.skipIf(os.name == 'nt', 'testConcurrentWorkflow skipped on Windows')
    def testConcurrentWorkflow(self):
        if False:
            i = 10
            return i + 15
        '\n        Test running concurrent task actions\n        '
        nop = Nop()
        workflow = Workflow([Task([nop, nop], concurrency='thread')])
        results = list(workflow([2, 4]))
        self.assertEqual(results, [(2, 2), (4, 4)])
        workflow = Workflow([Task([nop, nop], concurrency='process')])
        results = list(workflow([2, 4]))
        self.assertEqual(results, [(2, 2), (4, 4)])
        workflow = Workflow([Task([nop, nop], concurrency='unknown')])
        results = list(workflow([2, 4]))
        self.assertEqual(results, [(2, 2), (4, 4)])

    def testConsoleWorkflow(self):
        if False:
            i = 10
            return i + 15
        '\n        Test a console task\n        '
        workflow = Workflow([ConsoleTask()])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            list(workflow([{'id': 1, 'text': 'Sentence 1'}, {'id': 2, 'text': 'Sentence 2'}]))
        self.assertIn('Sentence 2', output.getvalue())

    def testExportWorkflow(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test an export task\n        '
        path = os.path.join(tempfile.gettempdir(), 'export.xlsx')
        workflow = Workflow([ExportTask(output=path)])
        list(workflow([{'id': 1, 'text': 'Sentence 1'}, {'id': 2, 'text': 'Sentence 2'}]))
        self.assertGreater(os.path.getsize(path), 0)
        path = os.path.join(tempfile.gettempdir(), 'export.csv')
        workflow = Workflow([ExportTask(output=path)])
        list(workflow([{'id': 1, 'text': 'Sentence 1'}, {'id': 2, 'text': 'Sentence 2'}]))
        self.assertGreater(os.path.getsize(path), 0)
        path = os.path.join(tempfile.gettempdir(), 'export-timestamp.csv')
        workflow = Workflow([ExportTask(output=path, timestamp=True)])
        list(workflow([{'id': 1, 'text': 'Sentence 1'}, {'id': 2, 'text': 'Sentence 2'}]))
        path = glob.glob(os.path.join(tempfile.gettempdir(), 'export-timestamp*.csv'))[0]
        self.assertGreater(os.path.getsize(path), 0)

    def testExtractWorkflow(self):
        if False:
            print('Hello World!')
        '\n        Test column extraction tasks\n        '
        workflow = Workflow([Task(lambda x: x, unpack=False, column=0)], batch=1)
        results = list(workflow([(0, 1)]))
        self.assertEqual(results[0], 0)
        results = list(workflow([(0, (1, 2), None)]))
        self.assertEqual(results[0], (0, 1, None))
        results = list(workflow([1]))
        self.assertEqual(results[0], 1)

    def testImageWorkflow(self):
        if False:
            return 10
        '\n        Test an image task\n        '
        workflow = Workflow([ImageTask()])
        results = list(workflow([Utils.PATH + '/books.jpg']))
        self.assertEqual(results[0].size, (1024, 682))

    def testInvalidWorkflow(self):
        if False:
            i = 10
            return i + 15
        '\n        Test task with invalid parameters\n        '
        with self.assertRaises(TypeError):
            Task(invalid=True)

    def testMergeWorkflow(self):
        if False:
            return 10
        '\n        Test merge tasks\n        '
        task = Task([lambda x: [pow(y, 2) for y in x], lambda x: [pow(y, 3) for y in x]], merge='hstack')
        workflow = Workflow([task])
        results = list(workflow([2, 4]))
        self.assertEqual(results, [(4, 8), (16, 64)])
        task.merge = 'vstack'
        results = list(workflow([2, 4]))
        self.assertEqual(results, [4, 8, 16, 64])
        task.merge = 'concat'
        results = list(workflow([2, 4]))
        self.assertEqual(results, ['4. 8', '16. 64'])
        task.merge = None
        results = list(workflow([2, 4, 6]))
        self.assertEqual(results, [[4, 16, 36], [8, 64, 216]])
        workflow = Workflow([Task(lambda x: [(0, y, None) for y in x])])
        results = list(workflow([(1, 'text', 'tags')]))
        self.assertEqual(results[0], (0, 'text', None))

    def testMergeUnbalancedWorkflow(self):
        if False:
            return 10
        '\n        Test merge tasks with unbalanced outputs (i.e. one action produce more output than another for same input).\n        '
        nop = Nop()
        segment1 = Segmentation(sentences=True)
        task = Task([nop, segment1])
        workflow = Workflow([task])
        results = list(workflow(['This is a test sentence. And another sentence to split.']))
        self.assertEqual(results, [('This is a test sentence. And another sentence to split.', ['This is a test sentence.', 'And another sentence to split.'])])
        task.merge = 'vstack'
        workflow = Workflow([task])
        results = list(workflow(['This is a test sentence. And another sentence to split.']))
        self.assertEqual(results, ['This is a test sentence. And another sentence to split.', 'This is a test sentence.', 'And another sentence to split.'])

    def testNumpyWorkflow(self):
        if False:
            print('Hello World!')
        '\n        Test a numpy workflow\n        '
        task = Task([lambda x: np.power(x, 2), lambda x: np.power(x, 3)], merge='hstack')
        workflow = Workflow([task])
        results = list(workflow(np.array([2, 4])))
        self.assertTrue(np.array_equal(np.array(results), np.array([[4, 8], [16, 64]])))
        task.merge = 'vstack'
        results = list(workflow(np.array([2, 4])))
        self.assertEqual(results, [4, 8, 16, 64])
        task.merge = None
        results = list(workflow(np.array([2, 4, 6])))
        self.assertTrue(np.array_equal(np.array(results), np.array([[4, 16, 36], [8, 64, 216]])))

    def testRetrieveWorkflow(self):
        if False:
            while True:
                i = 10
        '\n        Test a retrieve task\n        '
        workflow = Workflow([RetrieveTask()])
        results = list(workflow(['file://' + Utils.PATH + '/books.jpg']))
        self.assertTrue(results[0].endswith('books.jpg'))
        workflow = Workflow([RetrieveTask(directory=os.path.join(tempfile.gettempdir(), 'retrieve'))])
        results = list(workflow(['file://' + Utils.PATH + '/books.jpg']))
        self.assertTrue(results[0].endswith('books.jpg'))
        workflow = Workflow([RetrieveTask(flatten=False)])
        results = list(workflow(['file://' + Utils.PATH + '/books.jpg']))
        self.assertTrue(results[0].endswith('books.jpg') and 'txtai' in results[0])

    def testScheduleWorkflow(self):
        if False:
            print('Hello World!')
        '\n        Test workflow schedules\n        '
        workflow = Workflow([Task()])
        workflow.schedule('* * * * * *', ['test'], 1)
        self.assertEqual(len(workflow.tasks), 1)
        workflow = "\n        segmentation:\n            sentences: true\n        workflow:\n            segment:\n                schedule:\n                    cron: '* * * * * *'\n                    elements:\n                        - a sentence to segment\n                    iterations: 1\n                tasks:\n                    - action: segmentation\n                      task: console\n        "
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            app = API(workflow)
            app.wait()
        self.assertIn('a sentence to segment', output.getvalue())

    def testScheduleErrorWorkflow(self):
        if False:
            print('Hello World!')
        '\n        Test workflow schedules with errors\n        '

        def action(elements):
            if False:
                print('Hello World!')
            raise FileNotFoundError
        with self.assertLogs() as logs:
            workflow = Workflow([Task(action=action)])
            workflow.schedule('* * * * * *', ['test'], 1)
        self.assertIn('FileNotFoundError', ' '.join(logs.output))

    def testStorageWorkflow(self):
        if False:
            while True:
                i = 10
        '\n        Test a storage task\n        '
        workflow = Workflow([StorageTask()])
        results = list(workflow(['local://' + Utils.PATH, 'test string']))
        self.assertEqual(len(results), 19)

    def testTemplateInput(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test template task input\n        '
        workflow = Workflow([TemplateTask(template='This is a {text}')])
        results = list(workflow(['prompt']))
        self.assertEqual(results[0], 'This is a prompt')
        results = list(workflow([{'text': 'prompt'}]))
        self.assertEqual(results[0], 'This is a prompt')
        workflow = Workflow([TemplateTask(template='This is a {arg0}', unpack=False)])
        results = list(workflow([('prompt',)]))
        self.assertEqual(results[0], 'This is a prompt')
        with self.assertRaises(KeyError):
            workflow = Workflow([TemplateTask(template='No variables')])
            results = list(workflow([{'unused': 'prompt'}]))
        workflow = Workflow([TemplateTask()])
        results = list(workflow(['prompt']))
        self.assertEqual(results[0], 'prompt')

    def testTemplateRules(self):
        if False:
            print('Hello World!')
        '\n        Test template task rules\n        '
        workflow = Workflow([TemplateTask(template='This is a {text}', rules={'text': 'Test skip'})])
        results = list(workflow([{'text': 'Test skip'}]))
        self.assertEqual(results[0], 'Test skip')
        results = list(workflow([{'text': 'prompt'}]))
        self.assertEqual(results[0], 'This is a prompt')

    def testTemplateExtractor(self):
        if False:
            return 10
        '\n        Test extractor template task\n        '
        workflow = Workflow([ExtractorTask(template='This is a {text}')])
        results = list(workflow(['prompt']))
        self.assertEqual(results[0], {'query': 'prompt', 'question': 'This is a prompt'})
        workflow = Workflow([ExtractorTask(template='This is a {text}')])
        results = list(workflow([{'query': 'query', 'question': 'prompt'}]))
        self.assertEqual(results[0], {'query': 'query', 'question': 'This is a prompt'})
        workflow = Workflow([ExtractorTask(template='This is a {text} with another {param}')])
        results = list(workflow([{'query': 'query', 'question': 'prompt', 'param': 'value'}]))
        self.assertEqual(results[0], {'query': 'query', 'question': 'This is a prompt with another value', 'param': 'value'})

    def testTensorTransformWorkflow(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a tensor workflow with list transformations\n        '
        task = Task(lambda x: x.tolist())
        workflow = Workflow([task])
        results = list(workflow(np.array([2])))
        self.assertEqual(results, [2])
        task = Task(lambda x: [x.tolist() * 2])
        workflow = Workflow([task])
        results = list(workflow(np.array([2])))
        self.assertEqual(results, [2, 2])

    def testTorchWorkflow(self):
        if False:
            return 10
        '\n        Test a torch workflow\n        '
        task = Task([lambda x: torch.pow(x, 2), lambda x: torch.pow(x, 3)], merge='hstack')
        workflow = Workflow([task])
        results = np.array([x.numpy() for x in workflow(torch.tensor([2, 4]))])
        self.assertTrue(np.array_equal(results, np.array([[4, 8], [16, 64]])))
        task.merge = 'vstack'
        results = list(workflow(torch.tensor([2, 4])))
        self.assertEqual(results, [4, 8, 16, 64])
        task.merge = None
        results = np.array([x.numpy() for x in workflow(torch.tensor([2, 4, 6]))])
        self.assertTrue(np.array_equal(np.array(results), np.array([[4, 16, 36], [8, 64, 216]])))

    def testYamlFunctionWorkflow(self):
        if False:
            i = 10
            return i + 15
        '\n        Test YAML workflow with a function action\n        '

        def action(elements):
            if False:
                i = 10
                return i + 15
            return [x * 2 for x in elements]
        sys.modules[__name__].action = action
        workflow = '\n        workflow:\n            run:\n                tasks:\n                    - testworkflow.action\n        '
        app = API(workflow)
        self.assertEqual(list(app.workflow('run', [1, 2])), [2, 4])

    def testYamlIndexWorkflow(self):
        if False:
            while True:
                i = 10
        '\n        Test reading a YAML index workflow in Python.\n        '
        app = API(self.config)
        self.assertEqual(list(app.workflow('index', ['This is a test sentence. And another sentence to split.'])), ['This is a test sentence.', 'And another sentence to split.'])
        path = os.path.join(tempfile.gettempdir(), 'workflow.yml')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.config)
        app = API(path)
        self.assertEqual(list(app.workflow('index', ['This is a test sentence. And another sentence to split.'])), ['This is a test sentence.', 'And another sentence to split.'])
        app = API(API.read(self.config))
        self.assertEqual(list(app.workflow('index', ['This is a test sentence. And another sentence to split.'])), ['This is a test sentence.', 'And another sentence to split.'])

    def testYamlSearchWorkflow(self):
        if False:
            while True:
                i = 10
        '\n        Test reading a YAML search workflow in Python.\n        '
        app = API(self.config)
        list(app.workflow('index', ['This is a test sentence. And another sentence to split.']))
        self.assertEqual(list(app.workflow('search', ['another']))[0]['text'], 'And another sentence to split.')

    def testYamlWorkflowTask(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test YAML workflow with a workflow task\n        '

        def action(elements):
            if False:
                i = 10
                return i + 15
            return [x * 2 for x in elements]
        sys.modules[__name__].action = action
        workflow = '\n        workflow:\n            run:\n                tasks:\n                    - testworkflow.action\n            flow:\n                tasks:\n                    - run\n        '
        app = API(workflow)
        self.assertEqual(list(app.workflow('flow', [1, 2])), [2, 4])

    def testYamlTransformWorkflow(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test reading a YAML transform workflow in Python.\n        '
        app = API(self.config)
        self.assertEqual(len(list(app.workflow('transform', ['text']))[0]), 128)

    def testYamlError(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test reading a YAML workflow with errors.\n        '
        config = '\n        # Workflow definitions\n        workflow:\n            error:\n                tasks:\n                    - action: error\n        '
        with self.assertRaises(KeyError):
            API(config)
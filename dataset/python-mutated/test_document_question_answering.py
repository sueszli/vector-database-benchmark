import unittest
from datasets import load_dataset
from transformers import load_tool
from .test_tools_common import ToolTesterMixin

class DocumentQuestionAnsweringToolTester(unittest.TestCase, ToolTesterMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tool = load_tool('document-question-answering')
        self.tool.setup()
        self.remote_tool = load_tool('document-question-answering', remote=True)

    def test_exact_match_arg(self):
        if False:
            i = 10
            return i + 15
        dataset = load_dataset('hf-internal-testing/example-documents', split='test')
        document = dataset[0]['image']
        result = self.tool(document, 'When is the coffee break?')
        self.assertEqual(result, '11-14 to 11:39 a.m.')

    def test_exact_match_arg_remote(self):
        if False:
            print('Hello World!')
        dataset = load_dataset('hf-internal-testing/example-documents', split='test')
        document = dataset[0]['image']
        result = self.remote_tool(document, 'When is the coffee break?')
        self.assertEqual(result, '11-14 to 11:39 a.m.')

    def test_exact_match_kwarg(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = load_dataset('hf-internal-testing/example-documents', split='test')
        document = dataset[0]['image']
        self.tool(document=document, question='When is the coffee break?')

    def test_exact_match_kwarg_remote(self):
        if False:
            i = 10
            return i + 15
        dataset = load_dataset('hf-internal-testing/example-documents', split='test')
        document = dataset[0]['image']
        result = self.remote_tool(document=document, question='When is the coffee break?')
        self.assertEqual(result, '11-14 to 11:39 a.m.')
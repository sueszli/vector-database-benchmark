"""
LLM module tests
"""
import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from txtai.pipeline import LLM

class TestLLM(unittest.TestCase):
    """
    LLM tests.
    """

    def testArguments(self):
        if False:
            return 10
        '\n        Test pipeline keyword arguments\n        '
        start = 'Hello, how are'
        model = LLM('hf-internal-testing/tiny-random-gpt2', task='language-generation', torch_dtype='torch.float32')
        self.assertIsNotNone(model(start))
        model = LLM('hf-internal-testing/tiny-random-gpt2', task='language-generation', torch_dtype=torch.float32)
        self.assertIsNotNone(model(start))

    def testExternal(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test externally loaded model\n        '
        model = AutoModelForCausalLM.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        model = LLM((model, tokenizer), template='{text}')
        start = 'Hello, how are'
        self.assertIsNotNone(model(start))
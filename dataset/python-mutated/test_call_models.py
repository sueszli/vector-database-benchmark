from bigdl.llm.models import Llama, Bloom, Gptneox, Starcoder
from bigdl.llm.transformers import LlamaForCausalLM, BloomForCausalLM, GptneoxForCausalLM, StarcoderForCausalLM
import pytest
from unittest import TestCase
import os

class Test_Models_Basics(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.llama_model_path = os.environ.get('LLAMA_INT4_CKPT_PATH')
        self.bloom_model_path = os.environ.get('BLOOM_INT4_CKPT_PATH')
        self.gptneox_model_path = os.environ.get('GPTNEOX_INT4_CKPT_PATH')
        self.starcoder_model_path = os.environ.get('STARCODER_INT4_CKPT_PATH')
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2

    def test_llama_completion_success(self):
        if False:
            i = 10
            return i + 15
        llm = Llama(self.llama_model_path, n_threads=self.n_threads)
        output = llm('What is the capital of France?', max_tokens=32, stream=False)

    def test_llama_completion_with_stream_success(self):
        if False:
            for i in range(10):
                print('nop')
        llm = Llama(self.llama_model_path, n_threads=self.n_threads)
        output = llm('What is the capital of France?', max_tokens=32, stream=True)

    def test_llama_for_causallm(self):
        if False:
            while True:
                i = 10
        llm = LlamaForCausalLM.from_pretrained(self.llama_model_path, native=True, n_threads=self.n_threads)
        output = llm('What is the capital of France?', max_tokens=32, stream=False)

    def test_bloom_completion_success(self):
        if False:
            return 10
        llm = Bloom(self.bloom_model_path, n_threads=self.n_threads)
        output = llm('What is the capital of France?', max_tokens=32, stream=False)

    def test_bloom_completion_with_stream_success(self):
        if False:
            return 10
        llm = Bloom(self.bloom_model_path, n_threads=self.n_threads)
        output = llm('What is the capital of France?', max_tokens=32, stream=True)

    def test_bloom_for_causallm(self):
        if False:
            for i in range(10):
                print('nop')
        llm = BloomForCausalLM.from_pretrained(self.bloom_model_path, native=True, n_threads=self.n_threads)
        output = llm('What is the capital of France?', max_tokens=32, stream=False)

    def test_gptneox_completion_success(self):
        if False:
            i = 10
            return i + 15
        llm = Gptneox(self.gptneox_model_path, n_threads=self.n_threads)
        output = llm('Q: What is the capital of France? A:', max_tokens=32, stream=False)

    def test_gptneox_completion_with_stream_success(self):
        if False:
            i = 10
            return i + 15
        llm = Gptneox(self.gptneox_model_path, n_threads=self.n_threads)
        output = llm('Q: What is the capital of France? A:', max_tokens=32, stream=True)

    def test_getneox_for_causallm(self):
        if False:
            i = 10
            return i + 15
        llm = GptneoxForCausalLM.from_pretrained(self.gptneox_model_path, native=True, n_threads=self.n_threads)
        output = llm('Q: What is the capital of France? A:', max_tokens=32, stream=False)

    def test_starcoder_completion_success(self):
        if False:
            return 10
        llm = Starcoder(self.starcoder_model_path, n_threads=self.n_threads)
        output = llm('def print_hello_world(', max_tokens=32, stream=False)

    def test_starcoder_completion_with_stream_success(self):
        if False:
            return 10
        llm = Starcoder(self.starcoder_model_path, n_threads=self.n_threads)
        output = llm('def print_hello_world(', max_tokens=32, stream=True)

    def test_starcoder_for_causallm(self):
        if False:
            return 10
        llm = StarcoderForCausalLM.from_pretrained(self.starcoder_model_path, native=True, n_threads=self.n_threads)
        output = llm('def print_hello_world(', max_tokens=32, stream=False)
if __name__ == '__main__':
    pytest.main([__file__])
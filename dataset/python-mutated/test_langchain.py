from bigdl.llm.langchain.embeddings import *
from bigdl.llm.langchain.llms import *
import pytest
from unittest import TestCase
import os

class Test_Models_Basics(TestCase):

    def setUp(self):
        if False:
            return 10
        self.llama_model_path = os.environ.get('LLAMA_INT4_CKPT_PATH')
        self.bloom_model_path = os.environ.get('BLOOM_INT4_CKPT_PATH')
        self.gptneox_model_path = os.environ.get('GPTNEOX_INT4_CKPT_PATH')
        self.starcoder_model_path = os.environ.get('STARCODER_INT4_CKPT_PATH')
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2

    def test_langchain_llm_embedding_llama(self):
        if False:
            for i in range(10):
                print('nop')
        bigdl_embeddings = LlamaEmbeddings(model_path=self.llama_model_path)
        text = 'This is a test document.'
        query_result = bigdl_embeddings.embed_query(text)
        doc_result = bigdl_embeddings.embed_documents([text])

    def test_langchain_llm_embedding_gptneox(self):
        if False:
            while True:
                i = 10
        bigdl_embeddings = GptneoxEmbeddings(model_path=self.gptneox_model_path)
        text = 'This is a test document.'
        query_result = bigdl_embeddings.embed_query(text)
        doc_result = bigdl_embeddings.embed_documents([text])

    def test_langchain_llm_embedding_bloom(self):
        if False:
            print('Hello World!')
        bigdl_embeddings = BloomEmbeddings(model_path=self.bloom_model_path)
        text = 'This is a test document.'
        query_result = bigdl_embeddings.embed_query(text)
        doc_result = bigdl_embeddings.embed_documents([text])

    def test_langchain_llm_embedding_starcoder(self):
        if False:
            for i in range(10):
                print('nop')
        bigdl_embeddings = StarcoderEmbeddings(model_path=self.starcoder_model_path)
        text = 'This is a test document.'
        query_result = bigdl_embeddings.embed_query(text)
        doc_result = bigdl_embeddings.embed_documents([text])

    def test_langchain_llm_llama(self):
        if False:
            return 10
        llm = LlamaLLM(model_path=self.llama_model_path, max_tokens=32, n_threads=self.n_threads)
        question = 'What is AI?'
        result = llm(question)

    def test_langchain_llm_gptneox(self):
        if False:
            while True:
                i = 10
        llm = GptneoxLLM(model_path=self.gptneox_model_path, max_tokens=32, n_threads=self.n_threads)
        question = 'What is AI?'
        result = llm(question)

    def test_langchain_llm_bloom(self):
        if False:
            while True:
                i = 10
        llm = BloomLLM(model_path=self.bloom_model_path, max_tokens=32, n_threads=self.n_threads)
        question = 'What is AI?'
        result = llm(question)

    def test_langchain_llm_starcoder(self):
        if False:
            for i in range(10):
                print('nop')
        llm = StarcoderLLM(model_path=self.starcoder_model_path, max_tokens=32, n_threads=self.n_threads)
        question = 'What is AI?'
        result = llm(question)
if __name__ == '__main__':
    pytest.main([__file__])
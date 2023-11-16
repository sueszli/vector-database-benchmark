from bigdl.llm.langchain.llms import TransformersLLM, TransformersPipelineLLM, LlamaLLM, BloomLLM
from bigdl.llm.langchain.embeddings import TransformersEmbeddings, LlamaEmbeddings, BloomEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import pytest
from unittest import TestCase
import os

class Test_Langchain_Transformers_API(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.auto_model_path = os.environ.get('ORIGINAL_CHATGLM2_6B_PATH')
        self.auto_causal_model_path = os.environ.get('ORIGINAL_REPLIT_CODE_PATH')
        self.llama_model_path = os.environ.get('LLAMA_ORIGIN_PATH')
        self.bloom_model_path = os.environ.get('BLOOM_ORIGIN_PATH')
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2

    def test_pipeline_llm(self):
        if False:
            print('Hello World!')
        texts = 'def hello():\n  print("hello world")\n'
        bigdl_llm = TransformersPipelineLLM.from_model_id(model_id=self.auto_causal_model_path, task='text-generation', model_kwargs={'trust_remote_code': True})
        output = bigdl_llm(texts)
        res = 'hello()' in output
        self.assertTrue(res)

    def test_causalLM_embeddings(self):
        if False:
            while True:
                i = 10
        bigdl_embeddings = BloomEmbeddings(model_path=self.bloom_model_path, model_kwargs={'trust_remote_code': True}, native=False)
        text = 'This is a test document.'
        query_result = bigdl_embeddings.embed_query(text)
        doc_result = bigdl_embeddings.embed_documents([text])
        bigdl_llm = BloomLLM(model_path=self.bloom_model_path, model_kwargs={'trust_remote_code': True}, native=False)
        res = bigdl_llm(text)
    '\n    def test_transformers_llama_embeddings(self):\n        bigdl_embeddings = TransformersEmbeddings.from_model_id(model_id=self.llama_model_path, model_kwargs={\'trust_remote_code\': True})\n        text = "This is a test document."\n        query_result = bigdl_embeddings.embed_query(text)\n        doc_result = bigdl_embeddings.embed_documents([text])\n\n        bigdl_llm = TransformersLLM.from_model_id(model_id=self.llama_model_path, model_kwargs={\'trust_remote_code\': True})\n        res = bigdl_llm(text)\n    '

    def test_qa_chain(self):
        if False:
            return 10
        texts = '\n            AI is a machine’s ability to perform the cognitive functions \n            we associate with human minds, such as perceiving, reasoning, \n            learning, interacting with an environment, problem solving,\n            and even exercising creativity. You’ve probably interacted \n            with AI even if you didn’t realize it—voice assistants like Siri \n            and Alexa are founded on AI technology, as are some customer \n            service chatbots that pop up to help you navigate websites.\n            '
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(texts)
        query = 'What is AI?'
        embeddings = TransformersEmbeddings.from_model_id(model_id=self.auto_model_path, model_kwargs={'trust_remote_code': True})
        docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{'source': str(i)} for i in range(len(texts))]).as_retriever()
        docs = docsearch.get_relevant_documents(query)
        bigdl_llm = TransformersLLM.from_model_id(model_id=self.auto_model_path, model_kwargs={'trust_remote_code': True})
        doc_chain = load_qa_chain(bigdl_llm, chain_type='stuff', prompt=QA_PROMPT)
        output = doc_chain.run(input_documents=docs, question=query)
        res = 'AI' in output
        self.assertTrue(res)
    '\n    def test_qa_chain_causalLM(self):\n        texts = \'\'\'\n            AI is a machine’s ability to perform the cognitive functions \n            we associate with human minds, such as perceiving, reasoning, \n            learning, interacting with an environment, problem solving,\n            and even exercising creativity. You’ve probably interacted \n            with AI even if you didn’t realize it—voice assistants like Siri \n            and Alexa are founded on AI technology, as are some customer \n            service chatbots that pop up to help you navigate websites.\n            \'\'\'\n        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)\n        texts = text_splitter.split_text(texts)\n        query = \'What is AI?\'\n        embeddings = LlamaEmbeddings(model_path=self.llama_model_path, model_kwargs={\'trust_remote_code\': True}, native=False)\n\n        docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()\n\n        #get relavant texts\n        docs = docsearch.get_relevant_documents(query)\n        bigdl_llm = LlamaLLM(model_path=self.llama_model_path, model_kwargs={\'trust_remote_code\': True}, native=False)\n        doc_chain = load_qa_chain(bigdl_llm, chain_type="stuff", prompt=QA_PROMPT)\n        output = doc_chain.run(input_documents=docs, question=query)\n        res = "AI" in output\n        self.assertTrue(res)\n    '
if __name__ == '__main__':
    pytest.main([__file__])
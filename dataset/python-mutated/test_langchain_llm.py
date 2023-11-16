"""Unit tests for the base LLM class"""
from langchain.llms import OpenAI
import pytest
from pandasai.llm import LangchainLLM
from pandasai.prompts import AbstractPrompt
from unittest.mock import Mock

class TestLangchainLLM:
    """Unit tests for the LangChain wrapper LLM class"""

    @pytest.fixture
    def langchain_llm(self):
        if False:
            i = 10
            return i + 15

        class FakeOpenAI(OpenAI):
            openai_api_key = 'fake_key'

            def __call__(self, _prompt, stop=None, callbacks=None, **kwargs):
                if False:
                    print('Hello World!')
                return Mock(return_value='Custom response')()
        langchain_llm = FakeOpenAI()
        return langchain_llm

    @pytest.fixture
    def prompt(self):
        if False:
            return 10

        class MockAbstractPrompt(AbstractPrompt):
            template: str = 'Hello'
        return MockAbstractPrompt()

    def test_langchain_llm_type(self, langchain_llm):
        if False:
            i = 10
            return i + 15
        langchain_wrapper = LangchainLLM(langchain_llm)
        assert langchain_wrapper.type == 'langchain_openai'

    def test_langchain_model_call(self, langchain_llm, prompt):
        if False:
            for i in range(10):
                print('nop')
        langchain_wrapper = LangchainLLM(langchain_llm)
        assert langchain_wrapper.call(instruction=prompt, suffix='!') == 'Custom response'
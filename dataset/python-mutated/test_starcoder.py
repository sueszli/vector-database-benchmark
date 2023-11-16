"""Unit tests for the starcoder LLM class"""
import pytest
from pandasai.exceptions import APIKeyNotFoundError
from pandasai.llm import Starcoder

class TestStarcoderLLM:
    """Unit tests for the Starcoder LLM class"""

    def test_type(self):
        if False:
            for i in range(10):
                print('nop')
        assert Starcoder(api_token='test').type == 'starcoder'

    def test_init(self):
        if False:
            while True:
                i = 10
        with pytest.raises(APIKeyNotFoundError):
            Starcoder()
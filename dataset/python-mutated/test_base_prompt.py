"""Unit tests for the base prompt class"""
import pytest
from pandasai.prompts import AbstractPrompt

class TestBasePrompt:

    def test_instantiate_without_template(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError):
            AbstractPrompt()
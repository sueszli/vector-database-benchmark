import pytest
import os
from unittest import TestCase
import bigdl.nano.automl.hpo.space as space

class TestHPOSpace(TestCase):

    def test_categorical(self):
        if False:
            return 10
        choices = ['a', 'b', 'c', 'd']
        param = space.Categorical(*choices)
        assert param.cs
        assert param.rand in choices
        assert param.default in choices

    def test_real(self):
        if False:
            i = 10
            return i + 15
        min = 0.001
        max = 0.1
        param = space.Real(min, max)
        assert param.rand >= min and param.rand <= max
        assert param.default >= min and param.default <= max

    def test_int(self):
        if False:
            return 10
        min = 1
        max = 1000000.0
        param = space.Real(min, max)
        assert param.rand >= min and param.rand <= max
        assert param.default >= min and param.default <= max

    def test_list(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_dict(self):
        if False:
            return 10
        pass
if __name__ == '__main__':
    pytest.main([__file__])
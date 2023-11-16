import pytest
from unittest import TestCase
import bigdl.nano.automl as automl

class TestRegisterModules(TestCase):

    def test_register_layers(self):
        if False:
            return 10
        pass

    def test_register_activations(self):
        if False:
            return 10
        pass

    def test_register_tf_funcs(self):
        if False:
            i = 10
            return i + 15
        pass
if __name__ == '__main__':
    pytest.main([__file__])
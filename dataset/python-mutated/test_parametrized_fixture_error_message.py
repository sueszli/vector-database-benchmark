import unittest
import pytest

@pytest.fixture(params=[1, 2])
def two(request):
    if False:
        print('Hello World!')
    return request.param

@pytest.mark.usefixtures('two')
class TestSomethingElse(unittest.TestCase):

    def test_two(self):
        if False:
            return 10
        pass
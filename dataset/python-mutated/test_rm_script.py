import unittest
import pytest
from tests.run_command import run_command

@pytest.mark.slow
class TestRmScript(unittest.TestCase):

    def test_trash_put_works(self):
        if False:
            i = 10
            return i + 15
        result = run_command('.', 'trash-put')
        assert 'usage: trash-put [OPTION]... FILE...' in result.stderr.splitlines()

    def test_trash_put_touch_filesystem(self):
        if False:
            for i in range(10):
                print('nop')
        result = run_command('.', 'trash-put', ['non-existent'])
        assert "trash-put: cannot trash non existent 'non-existent'\n" == result.stderr
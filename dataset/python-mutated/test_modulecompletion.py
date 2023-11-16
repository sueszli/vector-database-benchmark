"""
Tests for module_completion.py
"""
import sys
import pytest
from spyder.utils.introspection.module_completion import get_preferred_submodules

@pytest.mark.skipif(sys.platform == 'darwin', reason="It's very slow on Mac")
def test_module_completion():
    if False:
        print('Hello World!')
    'Test module_completion.'
    assert 'numpy.linalg' in get_preferred_submodules()
if __name__ == '__main__':
    pytest.main()
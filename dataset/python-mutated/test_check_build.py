"""
Smoke Test the check_build module
"""
import pytest
from sklearn.__check_build import raise_build_error

def test_raise_build_error():
    if False:
        print('Hello World!')
    with pytest.raises(ImportError):
        raise_build_error(ImportError())
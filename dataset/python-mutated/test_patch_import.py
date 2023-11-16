from unittest.mock import MagicMock
import pytest
from tribler.core.utilities.patch_import import patch_import

@patch_import(['library_that_does_not_exist'])
def test_mock_import_mocked_lib():
    if False:
        while True:
            i = 10
    import library_that_does_not_exist
    assert library_that_does_not_exist
    assert library_that_does_not_exist.inner_function
    assert len(library_that_does_not_exist.inner_set) == 0

@patch_import('library_as_a_string')
def test_library_as_a_string():
    if False:
        for i in range(10):
            print('nop')
    import library_as_a_string
    assert library_as_a_string

@patch_import([])
def test_mock_import_import_real_lib():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ImportError):
        import library_that_does_not_exist
        library_that_does_not_exist.inner_function()

@patch_import(['time'])
def test_mock_import_not_strict():
    if False:
        while True:
            i = 10
    import time
    assert not isinstance(time, MagicMock)

@patch_import(['time'], strict=True)
def test_mock_import_strict():
    if False:
        for i in range(10):
            print('nop')
    import time
    assert isinstance(time, MagicMock)

@patch_import(['time'], always_raise_exception_on_import=True)
def test_mock_import_always_raise_exception_on_import():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ImportError):
        import time
        time.gmtime(0)
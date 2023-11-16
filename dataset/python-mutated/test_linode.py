"""
    :codeauthor: Nicole Thomas <nicole@saltstack.com>
"""
import pytest
from salt.cloud.clouds import linode

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {linode: {}}

def test_validate_name_first_character_invalid():
    if False:
        while True:
            i = 10
    '\n    Tests when name starts with an invalid character.\n    '
    assert linode._validate_name('-foo') is False
    assert linode._validate_name('_foo') is False

def test_validate_name_last_character_invalid():
    if False:
        print('Hello World!')
    '\n    Tests when name ends with an invalid character.\n    '
    assert linode._validate_name('foo-') is False
    assert linode._validate_name('foo_') is False

def test_validate_name_too_short():
    if False:
        return 10
    '\n    Tests when name has less than three letters.\n    '
    assert linode._validate_name('') is False
    assert linode._validate_name('ab') is False
    assert linode._validate_name('abc') is True

def test_validate_name_too_long():
    if False:
        return 10
    '\n    Tests when name has more than 48 letters.\n    '
    long_name = '1111-2222-3333-4444-5555-6666-7777-8888-9999-111'
    assert len(long_name) == 48
    assert linode._validate_name(long_name) is True
    long_name += '1'
    assert len(long_name) == 49
    assert linode._validate_name(long_name) is False

def test_validate_name_invalid_characters():
    if False:
        while True:
            i = 10
    '\n    Tests when name contains invalid characters.\n    '
    assert linode._validate_name('foo;bar') is False
    assert linode._validate_name('fooàààààbar') is False
    assert linode._validate_name('foo bar') is False

def test_validate_name_valid_characters():
    if False:
        print('Hello World!')
    '\n    Tests when name contains valid characters.\n    '
    assert linode._validate_name('foo123bar') is True
    assert linode._validate_name('foo-bar') is True
    assert linode._validate_name('foo_bar') is True
    assert linode._validate_name('1foo') is True
    assert linode._validate_name('foo0') is True
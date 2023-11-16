"""
Tests for the revised data collection consent mechanism in the cli/learning module.
"""
import pytest
from unittest.mock import patch
from pathlib import Path
from gpt_engineer.cli.learning import ask_collection_consent
from gpt_engineer.cli.learning import check_collection_consent

@pytest.fixture
def cleanup():
    if False:
        return 10
    yield
    if Path('.gpte_consent').exists():
        Path('.gpte_consent').unlink()
'\nTest the following 4 scenarios for check_collection_consent():\n    * The .gpte_consent file exists and its content is "true".\n    * The .gpte_consent file exists but its content is not "true".\n    * The .gpte_consent file does not exist and the user gives consent when asked.\n    * The .gpte_consent file does not exist and the user does not give consent when asked.\n'

def test_check_consent_file_exists_and_true(cleanup):
    if False:
        i = 10
        return i + 15
    Path('.gpte_consent').write_text('true')
    assert check_collection_consent() == True

def test_check_consent_file_exists_and_false(cleanup):
    if False:
        while True:
            i = 10
    Path('.gpte_consent').write_text('false')
    with patch('builtins.input', side_effect=['n']):
        assert check_collection_consent() == False

def test_check_consent_file_not_exists_and_user_says_yes(cleanup):
    if False:
        for i in range(10):
            print('nop')
    with patch('builtins.input', side_effect=['y']):
        assert check_collection_consent() == True
    assert Path('.gpte_consent').exists()
    assert Path('.gpte_consent').read_text() == 'true'

def test_check_consent_file_not_exists_and_user_says_no(cleanup):
    if False:
        print('Hello World!')
    with patch('builtins.input', side_effect=['n']):
        assert check_collection_consent() == False
    assert not Path('.gpte_consent').exists()
'\nTest the following 4 scenarios for ask_collection_consent():\n    1. The user immediately gives consent with "y":\n        * The .gpte_consent file is created with content "true".\n        * The function returns True.\n    2. The user immediately denies consent with "n":\n        * The .gpte_consent file is not created.\n        * The function returns False.\n    3. The user first provides an invalid response, then gives consent with "y":\n        * The user is re-prompted after the invalid input.\n        * The .gpte_consent file is created with content "true".\n        * The function returns True.\n    4. The user first provides an invalid response, then denies consent with "n":\n        * The user is re-prompted after the invalid input.\n        * The .gpte_consent file is not created.\n        * The function returns False.\n'

def test_ask_collection_consent_yes(cleanup):
    if False:
        while True:
            i = 10
    with patch('builtins.input', side_effect=['y']):
        result = ask_collection_consent()
    assert Path('.gpte_consent').exists()
    assert Path('.gpte_consent').read_text() == 'true'
    assert result == True

def test_ask_collection_consent_no(cleanup):
    if False:
        while True:
            i = 10
    with patch('builtins.input', side_effect=['n']):
        result = ask_collection_consent()
    assert not Path('.gpte_consent').exists()
    assert result == False

def test_ask_collection_consent_invalid_then_yes(cleanup):
    if False:
        print('Hello World!')
    with patch('builtins.input', side_effect=['invalid', 'y']):
        result = ask_collection_consent()
    assert Path('.gpte_consent').exists()
    assert Path('.gpte_consent').read_text() == 'true'
    assert result == True

def test_ask_collection_consent_invalid_then_no(cleanup):
    if False:
        while True:
            i = 10
    with patch('builtins.input', side_effect=['invalid', 'n']):
        result = ask_collection_consent()
    assert not Path('.gpte_consent').exists()
    assert result == False
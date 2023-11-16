"""
Tests for _clean_dir function
"""
import pytest
import salt.states.file as file
pytestmark = [pytest.mark.windows_whitelisted]

def test_normal():
    if False:
        print('Hello World!')
    expected = []
    result = file._clean_dir(root='/tmp/parent', keep=['/tmp/parent/meh-1.txt', '/tmp/parent/meh-2.txt'], exclude_pat=None)
    assert result == expected

def test_win_forward_slash():
    if False:
        i = 10
        return i + 15
    expected = []
    result = file._clean_dir(root='C:/test/parent', keep=['C:/test/parent/meh-1.txt', 'C:/test/parent/meh-2.txt'], exclude_pat=None)
    assert result == expected

def test_win_forward_slash_mixed_case():
    if False:
        i = 10
        return i + 15
    expected = []
    result = file._clean_dir(root='C:/test/parent', keep=['C:/test/parent/meh-1.txt', 'C:/test/Parent/Meh-2.txt'], exclude_pat=None)
    assert result == expected

def test_win_back_slash():
    if False:
        while True:
            i = 10
    expected = []
    result = file._clean_dir(root='C:\\test\\parent', keep=['C:\\test\\parent\\meh-1.txt', 'C:\\test\\parent\\meh-2.txt'], exclude_pat=None)
    assert result == expected

def test_win_back_slash_mixed_cased():
    if False:
        print('Hello World!')
    expected = []
    result = file._clean_dir(root='C:\\test\\parent', keep=['C:\\test\\parent\\meh-1.txt', 'C:\\test\\Parent\\Meh-2.txt'], exclude_pat=None)
    assert result == expected
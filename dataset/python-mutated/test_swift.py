"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.swift
"""
import pytest
import salt.modules.swift as swift
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {swift: {}}

def test_delete():
    if False:
        i = 10
        return i + 15
    '\n    Test for delete a container, or delete an object from a container.\n    '
    with patch.object(swift, '_auth', MagicMock()):
        assert swift.delete('mycontainer')
        assert swift.delete('mycontainer', path='myfile.png')

def test_get():
    if False:
        while True:
            i = 10
    '\n    Test for list the contents of a container,\n    or return an object from a container.\n    '
    with patch.object(swift, '_auth', MagicMock()):
        assert swift.get()
        assert swift.get('mycontainer')
        assert swift.get('mycontainer', path='myfile.png', return_bin=True)
        assert swift.get('mycontainer', path='myfile.png', local_file='/tmp/myfile.png')
        assert not swift.get('mycontainer', path='myfile.png')

def test_put():
    if False:
        i = 10
        return i + 15
    '\n    Test for create a new container, or upload an object to a container.\n    '
    with patch.object(swift, '_auth', MagicMock()):
        assert swift.put('mycontainer')
        assert swift.put('mycontainer', path='myfile.png', local_file='/tmp/myfile.png')
        assert not swift.put('mycontainer', path='myfile.png')
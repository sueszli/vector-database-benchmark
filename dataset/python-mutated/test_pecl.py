"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.pecl
"""
import pytest
import salt.modules.pecl as pecl
from tests.support.mock import patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {pecl: {}}

def test_install():
    if False:
        while True:
            i = 10
    '\n    Test to installs one or several pecl extensions.\n    '
    with patch.object(pecl, '_pecl', return_value='A'):
        assert pecl.install('fuse', force=True) == 'A'
        assert not pecl.install('fuse')
        with patch.object(pecl, 'list_', return_value={'A': ['A', 'B']}):
            assert pecl.install(['A', 'B'])

def test_uninstall():
    if False:
        return 10
    '\n    Test to uninstall one or several pecl extensions.\n    '
    with patch.object(pecl, '_pecl', return_value='A'):
        assert pecl.uninstall('fuse') == 'A'

def test_update():
    if False:
        i = 10
        return i + 15
    '\n    Test to update one or several pecl extensions.\n    '
    with patch.object(pecl, '_pecl', return_value='A'):
        assert pecl.update('fuse') == 'A'

def test_list_():
    if False:
        print('Hello World!')
    '\n    Test to list installed pecl extensions.\n    '
    with patch.object(pecl, '_pecl', return_value='A\nB'):
        assert pecl.list_('channel') == {}
"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import pytest
import salt.modules.cpan as cpan
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {cpan: {}}

def test_install():
    if False:
        i = 10
        return i + 15
    '\n    Test if it install a module from cpan\n    '
    mock = MagicMock(return_value='')
    with patch.dict(cpan.__salt__, {'cmd.run': mock}):
        mock = MagicMock(side_effect=[{'installed version': None}, {'installed version': '3.1'}])
        with patch.object(cpan, 'show', mock):
            assert cpan.install('Alloy') == {'new': '3.1', 'old': None}

def test_install_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it install a module from cpan\n    '
    mock = MagicMock(return_value="don't know what it is")
    with patch.dict(cpan.__salt__, {'cmd.run': mock}):
        assert cpan.install('Alloy') == {'error': 'CPAN cannot identify this package', 'new': None, 'old': None}

def test_remove():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it remove a module using cpan\n    '
    with patch('os.listdir', MagicMock(return_value=[''])):
        mock = MagicMock(return_value='')
        with patch.dict(cpan.__salt__, {'cmd.run': mock}):
            mock = MagicMock(return_value={'installed version': '2.1', 'cpan build dirs': [''], 'installed file': '/root'})
            with patch.object(cpan, 'show', mock):
                assert cpan.remove('Alloy') == {'new': None, 'old': '2.1'}

def test_remove_unexist_error():
    if False:
        print('Hello World!')
    '\n    Test if it try to remove an unexist module using cpan\n    '
    mock = MagicMock(return_value="don't know what it is")
    with patch.dict(cpan.__salt__, {'cmd.run': mock}):
        assert cpan.remove('Alloy') == {'error': 'This package does not seem to exist'}

def test_remove_noninstalled_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it remove non installed module using cpan\n    '
    mock = MagicMock(return_value={'installed version': None})
    with patch.object(cpan, 'show', mock):
        assert cpan.remove('Alloy') == {'new': None, 'old': None}

def test_remove_nopan_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it gives no cpan error while removing\n    '
    ret = {'error': 'No CPAN data available to use for uninstalling'}
    mock = MagicMock(return_value={'installed version': '2.1'})
    with patch.object(cpan, 'show', mock):
        assert cpan.remove('Alloy') == ret

def test_list():
    if False:
        i = 10
        return i + 15
    '\n    Test if it list installed Perl module\n    '
    mock = MagicMock(return_value='')
    with patch.dict(cpan.__salt__, {'cmd.run': mock}):
        assert cpan.list_() == {}

def test_show():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it show information about a specific Perl module\n    '
    mock = MagicMock(return_value='')
    with patch.dict(cpan.__salt__, {'cmd.run': mock}):
        assert cpan.show('Alloy') == {'error': 'This package does not seem to exist', 'name': 'Alloy'}

def test_show_mock():
    if False:
        while True:
            i = 10
    '\n    Test if it show information about a specific Perl module\n    '
    with patch('salt.modules.cpan.show', MagicMock(return_value={'Salt': 'salt'})):
        mock = MagicMock(return_value='Salt module installed')
        with patch.dict(cpan.__salt__, {'cmd.run': mock}):
            assert cpan.show('Alloy') == {'Salt': 'salt'}

def test_show_config():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it return a dict of CPAN configuration values\n    '
    mock = MagicMock(return_value='')
    with patch.dict(cpan.__salt__, {'cmd.run': mock}):
        assert cpan.show_config() == {}
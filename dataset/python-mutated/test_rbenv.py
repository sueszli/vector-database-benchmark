"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.rbenv
"""
import os
import pytest
import salt.modules.rbenv as rbenv
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {rbenv: {}}

def test_install():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for install Rbenv systemwide\n    '
    with patch.object(rbenv, '_rbenv_path', return_value=True):
        with patch.object(rbenv, '_install_rbenv', return_value=True):
            with patch.object(rbenv, '_install_ruby_build', return_value=True):
                with patch.object(os.path, 'expanduser', return_value='A'):
                    assert rbenv.install()

def test_update():
    if False:
        while True:
            i = 10
    '\n    Test for updates the current versions of Rbenv and Ruby-Build\n    '
    with patch.object(rbenv, '_rbenv_path', return_value=True):
        with patch.object(rbenv, '_update_rbenv', return_value=True):
            with patch.object(rbenv, '_update_ruby_build', return_value=True):
                with patch.object(os.path, 'expanduser', return_value='A'):
                    assert rbenv.update()

def test_is_installed():
    if False:
        i = 10
        return i + 15
    '\n    Test for check if Rbenv is installed.\n    '
    with patch.object(rbenv, '_rbenv_bin', return_value='A'):
        with patch.dict(rbenv.__salt__, {'cmd.has_exec': MagicMock(return_value=True)}):
            assert rbenv.is_installed()

def test_install_ruby():
    if False:
        return 10
    '\n    Test for install a ruby implementation.\n    '
    with patch.dict(rbenv.__grains__, {'os': 'FreeBSD'}):
        with patch.dict(rbenv.__salt__, {'config.get': MagicMock(return_value='True')}):
            with patch.object(rbenv, '_rbenv_exec', return_value={'retcode': 0, 'stderr': 'stderr'}):
                with patch.object(rbenv, 'rehash', return_value=None):
                    assert rbenv.install_ruby('ruby') == 'stderr'
            with patch.object(rbenv, '_rbenv_exec', return_value={'retcode': 1, 'stderr': 'stderr'}):
                with patch.object(rbenv, 'uninstall_ruby', return_value=None):
                    assert not rbenv.install_ruby('ruby')

def test_uninstall_ruby():
    if False:
        return 10
    '\n    Test for uninstall a ruby implementation.\n    '
    with patch.object(rbenv, '_rbenv_exec', return_value=None):
        assert rbenv.uninstall_ruby('ruby', 'runas')

def test_versions():
    if False:
        return 10
    '\n    Test for list the installed versions of ruby.\n    '
    with patch.object(rbenv, '_rbenv_exec', return_value='A\nBC\nD'):
        assert rbenv.versions() == ['A', 'BC', 'D']

def test_default():
    if False:
        print('Hello World!')
    '\n    Test for returns or sets the currently defined default ruby.\n    '
    with patch.object(rbenv, '_rbenv_exec', MagicMock(side_effect=[None, False])):
        assert rbenv.default('ruby', 'runas')
        assert rbenv.default() == ''

def test_list_():
    if False:
        while True:
            i = 10
    '\n    Test for list the installable versions of ruby.\n    '
    with patch.object(rbenv, '_rbenv_exec', return_value='A\nB\nCD\n'):
        assert rbenv.list_() == ['A', 'B', 'CD']

def test_rehash():
    if False:
        i = 10
        return i + 15
    '\n    Test for run rbenv rehash to update the installed shims.\n    '
    with patch.object(rbenv, '_rbenv_exec', return_value=None):
        assert rbenv.rehash()

def test_do_with_ruby():
    if False:
        i = 10
        return i + 15
    "\n    Test for execute a ruby command with rbenv's shims using a\n    specific ruby version.\n    "
    with patch.object(rbenv, 'do', return_value='A'):
        assert rbenv.do_with_ruby('ruby', 'cmdline') == 'A'
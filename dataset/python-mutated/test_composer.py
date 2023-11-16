"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>
"""
import pytest
import salt.modules.composer as composer
from salt.exceptions import CommandExecutionError, CommandNotFoundError, SaltInvocationError
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {composer: {}}

def test_install():
    if False:
        print('Hello World!')
    '\n    Test for Install composer dependencies for a directory.\n    '
    mock = MagicMock(return_value=False)
    with patch.object(composer, '_valid_composer', mock):
        pytest.raises(CommandNotFoundError, composer.install, 'd')
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        pytest.raises(SaltInvocationError, composer.install, None)
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        mock = MagicMock(return_value={'retcode': 1, 'stderr': 'A'})
        with patch.dict(composer.__salt__, {'cmd.run_all': mock}):
            pytest.raises(CommandExecutionError, composer.install, 'd')
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        mock = MagicMock(return_value={'retcode': 0, 'stderr': 'A'})
        with patch.dict(composer.__salt__, {'cmd.run_all': mock}):
            assert composer.install('dir', None, None, None, None, None, None, None, None, None, True)
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        rval = {'retcode': 0, 'stderr': 'A', 'stdout': 'B'}
        mock = MagicMock(return_value=rval)
        with patch.dict(composer.__salt__, {'cmd.run_all': mock}):
            assert composer.install('dir') == rval

def test_update():
    if False:
        while True:
            i = 10
    '\n    Test for Update composer dependencies for a directory.\n    '
    mock = MagicMock(return_value=False)
    with patch.object(composer, '_valid_composer', mock):
        pytest.raises(CommandNotFoundError, composer.update, 'd')
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        mock = MagicMock(return_value=True)
        with patch.object(composer, 'did_composer_install', mock):
            pytest.raises(SaltInvocationError, composer.update, None)
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        mock = MagicMock(return_value=True)
        with patch.object(composer, 'did_composer_install', mock):
            mock = MagicMock(return_value={'retcode': 1, 'stderr': 'A'})
            with patch.dict(composer.__salt__, {'cmd.run_all': mock}):
                pytest.raises(CommandExecutionError, composer.update, 'd')
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        mock = MagicMock(return_value=True)
        with patch.object(composer, 'did_composer_install', mock):
            mock = MagicMock(return_value={'retcode': 0, 'stderr': 'A'})
            with patch.dict(composer.__salt__, {'cmd.run_all': mock}):
                assert composer.update('dir', None, None, None, None, None, None, None, None, None, True)
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        mock = MagicMock(return_value=False)
        with patch.object(composer, 'did_composer_install', mock):
            mock = MagicMock(return_value={'retcode': 0, 'stderr': 'A'})
            with patch.dict(composer.__salt__, {'cmd.run_all': mock}):
                assert composer.update('dir', None, None, None, None, None, None, None, None, None, True)
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        mock = MagicMock(return_value=True)
        with patch.object(composer, 'did_composer_install', mock):
            rval = {'retcode': 0, 'stderr': 'A', 'stdout': 'B'}
            mock = MagicMock(return_value=rval)
            with patch.dict(composer.__salt__, {'cmd.run_all': mock}):
                assert composer.update('dir') == rval
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        mock = MagicMock(return_value=False)
        with patch.object(composer, 'did_composer_install', mock):
            rval = {'retcode': 0, 'stderr': 'A', 'stdout': 'B'}
            mock = MagicMock(return_value=rval)
            with patch.dict(composer.__salt__, {'cmd.run_all': mock}):
                assert composer.update('dir') == rval

def test_selfupdate():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Composer selfupdate\n    '
    mock = MagicMock(return_value=False)
    with patch.object(composer, '_valid_composer', mock):
        pytest.raises(CommandNotFoundError, composer.selfupdate)
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        mock = MagicMock(return_value={'retcode': 1, 'stderr': 'A'})
        with patch.dict(composer.__salt__, {'cmd.run_all': mock}):
            pytest.raises(CommandExecutionError, composer.selfupdate)
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        mock = MagicMock(return_value={'retcode': 0, 'stderr': 'A'})
        with patch.dict(composer.__salt__, {'cmd.run_all': mock}):
            assert composer.selfupdate(quiet=True)
    mock = MagicMock(return_value=True)
    with patch.object(composer, '_valid_composer', mock):
        rval = {'retcode': 0, 'stderr': 'A', 'stdout': 'B'}
        mock = MagicMock(return_value=rval)
        with patch.dict(composer.__salt__, {'cmd.run_all': mock}):
            assert composer.selfupdate() == rval
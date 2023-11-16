import logging
import os
import sys
from unittest.mock import Mock
import pytest
import apprise
from apprise.plugins.NotifyMacOSX import NotifyMacOSX
from helpers import reload_plugin
logging.disable(logging.CRITICAL)
if sys.platform not in ['darwin', 'linux']:
    pytest.skip('Only makes sense on macOS, but also works on Linux', allow_module_level=True)

@pytest.fixture
def pretend_macos(mocker):
    if False:
        while True:
            i = 10
    '\n    Fixture to simulate a macOS environment.\n    '
    mocker.patch('platform.system', return_value='Darwin')
    mocker.patch('platform.mac_ver', return_value=('10.8', ('', '', ''), ''))
    current_module = sys.modules[__name__]
    reload_plugin('NotifyMacOSX', replace_in=current_module)

@pytest.fixture
def terminal_notifier(mocker, tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    Fixture for providing a surrogate for the `terminal-notifier` program.\n    '
    notifier_program = tmp_path.joinpath('terminal-notifier')
    notifier_program.write_text('#!/bin/sh\n\necho hello')
    os.chmod(notifier_program, 493)
    mocker.patch('apprise.plugins.NotifyMacOSX.NotifyMacOSX.notify_paths', (str(notifier_program),))
    yield notifier_program

@pytest.fixture
def macos_notify_environment(pretend_macos, terminal_notifier):
    if False:
        while True:
            i = 10
    "\n    Fixture to bundle general test case setup.\n\n    Use this fixture if you don't need access to the individual members.\n    "
    pass

def test_plugin_macosx_general_success(macos_notify_environment):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyMacOSX() general checks\n    '
    obj = apprise.Apprise.instantiate('macosx://_/?image=True', suppress_exceptions=False)
    assert isinstance(obj, NotifyMacOSX) is True
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    assert obj.notify(title='', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('macosx://_/?image=True', suppress_exceptions=False)
    assert isinstance(obj, NotifyMacOSX) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('macosx://_/?image=False', suppress_exceptions=False)
    assert isinstance(obj, NotifyMacOSX) is True
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('macosx://_/?sound=default', suppress_exceptions=False)
    assert isinstance(obj, NotifyMacOSX) is True
    assert obj.sound == 'default'
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True
    obj = apprise.Apprise.instantiate('macosx://_/?click=http://google.com', suppress_exceptions=False)
    assert isinstance(obj, NotifyMacOSX) is True
    assert obj.click == 'http://google.com'
    assert isinstance(obj.url(), str) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is True

def test_plugin_macosx_terminal_notifier_not_executable(pretend_macos, terminal_notifier):
    if False:
        i = 10
        return i + 15
    '\n    When the `terminal-notifier` program is inaccessible or not executable,\n    we are unable to send notifications.\n    '
    obj = apprise.Apprise.instantiate('macosx://', suppress_exceptions=False)
    os.chmod(terminal_notifier, 420)
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is False

def test_plugin_macosx_terminal_notifier_invalid(macos_notify_environment):
    if False:
        return 10
    '\n    When the `terminal-notifier` program is wrongly addressed,\n    notifications should fail.\n    '
    obj = apprise.Apprise.instantiate('macosx://', suppress_exceptions=False)
    obj.notify_path = 'invalid_missing-file'
    assert not os.path.isfile(obj.notify_path)
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is False

def test_plugin_macosx_terminal_notifier_croaks(mocker, macos_notify_environment):
    if False:
        i = 10
        return i + 15
    '\n    When the `terminal-notifier` program croaks on execution,\n    notifications should fail.\n    '
    mocker.patch('subprocess.Popen', return_value=Mock(returncode=1))
    obj = apprise.Apprise.instantiate('macosx://', suppress_exceptions=False)
    assert isinstance(obj, NotifyMacOSX) is True
    assert obj.notify(title='title', body='body', notify_type=apprise.NotifyType.INFO) is False

def test_plugin_macosx_pretend_linux(mocker, pretend_macos):
    if False:
        for i in range(10):
            print('nop')
    '\n    The notification object is disabled when pretending to run on Linux.\n    '
    mocker.patch('platform.system', return_value='Linux')
    reload_plugin('NotifyMacOSX')
    obj = apprise.Apprise.instantiate('macosx://', suppress_exceptions=False)
    assert obj is None

@pytest.mark.parametrize('macos_version', ['9.12', '10.7'])
def test_plugin_macosx_pretend_old_macos(mocker, macos_version):
    if False:
        i = 10
        return i + 15
    '\n    The notification object is disabled when pretending to run on older macOS.\n    '
    mocker.patch('platform.mac_ver', return_value=(macos_version, ('', '', ''), ''))
    reload_plugin('NotifyMacOSX')
    obj = apprise.Apprise.instantiate('macosx://', suppress_exceptions=False)
    assert obj is None
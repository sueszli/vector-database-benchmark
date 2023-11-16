from unittest.mock import MagicMock, patch
import pytest
from cura.API import Account
from cura.API.Account import SyncState
from cura.OAuth2.Models import UserProfile

@pytest.fixture()
def user_profile():
    if False:
        for i in range(10):
            print('nop')
    result = UserProfile()
    result.username = 'username!'
    result.profile_image_url = 'profile_image_url!'
    result.user_id = 'user_id!'
    return result

def test_login():
    if False:
        i = 10
        return i + 15
    account = Account(MagicMock())
    mocked_auth_service = MagicMock()
    account._authorization_service = mocked_auth_service
    account.logout = MagicMock()
    account.login()
    mocked_auth_service.startAuthorizationFlow.assert_called_once_with(False)
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance'):
        account._onLoginStateChanged(True)
    account.login()
    mocked_auth_service.startAuthorizationFlow.assert_called_once_with(False)
    account.login(force_logout_before_login=True)
    account.logout.assert_called_once_with()
    mocked_auth_service.startAuthorizationFlow.assert_called_with(True)
    assert mocked_auth_service.startAuthorizationFlow.call_count == 2

def test_initialize():
    if False:
        for i in range(10):
            print('nop')
    account = Account(MagicMock())
    mocked_auth_service = MagicMock()
    account._authorization_service = mocked_auth_service
    account.initialize()
    mocked_auth_service.loadAuthDataFromPreferences.assert_called_once_with()

def test_logout():
    if False:
        while True:
            i = 10
    account = Account(MagicMock())
    mocked_auth_service = MagicMock()
    account._authorization_service = mocked_auth_service
    account.logout()
    mocked_auth_service.deleteAuthData.assert_not_called()
    assert not account.isLoggedIn
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance'):
        account._onLoginStateChanged(True)
    assert account.isLoggedIn
    account.logout()
    mocked_auth_service.deleteAuthData.assert_called_once_with()

@patch('UM.Application.Application.getInstance')
def test_errorLoginState(application):
    if False:
        return 10
    account = Account(MagicMock())
    mocked_auth_service = MagicMock()
    account._authorization_service = mocked_auth_service
    account.loginStateChanged = MagicMock()
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance'):
        account._onLoginStateChanged(True, 'BLARG!')
    account.loginStateChanged.emit.called_with(False)
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance'):
        account._onLoginStateChanged(True)
        account._onLoginStateChanged(False, 'OMGZOMG!')
    account.loginStateChanged.emit.called_with(False)

def test_sync_success():
    if False:
        i = 10
        return i + 15
    account = Account(MagicMock())
    service1 = 'test_service1'
    service2 = 'test_service2'
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance'):
        account.setSyncState(service1, SyncState.SYNCING)
        assert account.syncState == SyncState.SYNCING
        account.setSyncState(service2, SyncState.SYNCING)
        assert account.syncState == SyncState.SYNCING
        account.setSyncState(service1, SyncState.SUCCESS)
        assert account.syncState == SyncState.SYNCING
        account.setSyncState(service2, SyncState.SUCCESS)
        assert account.syncState == SyncState.SUCCESS

def test_sync_update_action():
    if False:
        for i in range(10):
            print('nop')
    account = Account(MagicMock())
    service1 = 'test_service1'
    mockUpdateCallback = MagicMock()
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance'):
        account.setSyncState(service1, SyncState.SYNCING)
        assert account.syncState == SyncState.SYNCING
        account.setUpdatePackagesAction(mockUpdateCallback)
        account.onUpdatePackagesClicked()
        mockUpdateCallback.assert_called_once_with()
        account.setSyncState(service1, SyncState.SUCCESS)
        account.sync()
        account.setSyncState(service1, SyncState.SYNCING)
        assert account.syncState == SyncState.SYNCING
        account.onUpdatePackagesClicked()
        mockUpdateCallback.assert_called_once_with()
        assert account.updatePackagesEnabled is False
        account.setSyncState(service1, SyncState.SUCCESS)
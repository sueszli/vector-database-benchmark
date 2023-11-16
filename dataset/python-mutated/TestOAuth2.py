from datetime import datetime
from unittest.mock import MagicMock, Mock, patch
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtNetwork import QNetworkReply
from UM.Preferences import Preferences
from cura.OAuth2.AuthorizationHelpers import AuthorizationHelpers, TOKEN_TIMESTAMP_FORMAT
from cura.OAuth2.AuthorizationService import AuthorizationService, MYCLOUD_LOGOFF_URL
from cura.OAuth2.LocalAuthorizationServer import LocalAuthorizationServer
from cura.OAuth2.Models import OAuth2Settings, AuthenticationResponse, UserProfile
CALLBACK_PORT = 32118
OAUTH_ROOT = 'https://account.ultimaker.com'
CLOUD_API_ROOT = 'https://api.ultimaker.com'
OAUTH_SETTINGS = OAuth2Settings(OAUTH_SERVER_URL=OAUTH_ROOT, CALLBACK_PORT=CALLBACK_PORT, CALLBACK_URL='http://localhost:{}/callback'.format(CALLBACK_PORT), CLIENT_ID='', CLIENT_SCOPES='', AUTH_DATA_PREFERENCE_KEY='test/auth_data', AUTH_SUCCESS_REDIRECT='{}/app/auth-success'.format(OAUTH_ROOT), AUTH_FAILED_REDIRECT='{}/app/auth-error'.format(OAUTH_ROOT))
FAILED_AUTH_RESPONSE = AuthenticationResponse(success=False, err_message='FAILURE!')
SUCCESSFUL_AUTH_RESPONSE = AuthenticationResponse(access_token='beep', refresh_token='beep?', received_at=datetime.now().strftime(TOKEN_TIMESTAMP_FORMAT), expires_in=300, success=True)
EXPIRED_AUTH_RESPONSE = AuthenticationResponse(access_token='expired', refresh_token='beep?', received_at=datetime.now().strftime(TOKEN_TIMESTAMP_FORMAT), expires_in=300, success=True)
NO_REFRESH_AUTH_RESPONSE = AuthenticationResponse(access_token='beep', received_at=datetime.now().strftime(TOKEN_TIMESTAMP_FORMAT), expires_in=300, success=True)
MALFORMED_AUTH_RESPONSE = AuthenticationResponse(success=False)

def test_cleanAuthService() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Ensure that when setting up an AuthorizationService, no data is set.\n    '
    authorization_service = AuthorizationService(OAUTH_SETTINGS, Preferences())
    authorization_service.initialize()
    mock_callback = Mock()
    authorization_service.getUserProfile(mock_callback)
    mock_callback.assert_called_once_with(None)
    assert authorization_service.getAccessToken() is None

def test_refreshAccessTokenSuccess():
    if False:
        return 10
    authorization_service = AuthorizationService(OAUTH_SETTINGS, Preferences())
    authorization_service.initialize()
    with patch.object(AuthorizationService, 'getUserProfile', return_value=UserProfile()):
        authorization_service._storeAuthData(SUCCESSFUL_AUTH_RESPONSE)
    authorization_service.onAuthStateChanged.emit = MagicMock()
    with patch.object(AuthorizationHelpers, 'getAccessTokenUsingRefreshToken', return_value=SUCCESSFUL_AUTH_RESPONSE):
        authorization_service.refreshAccessToken()
        assert authorization_service.onAuthStateChanged.emit.called_with(True)

def test__parseJWTNoRefreshToken():
    if False:
        return 10
    '\n    Tests parsing the user profile if there is no refresh token stored, but there is a normal authentication token.\n\n    The request for the user profile using the authentication token should still work normally.\n    '
    authorization_service = AuthorizationService(OAUTH_SETTINGS, Preferences())
    with patch.object(AuthorizationService, 'getUserProfile', return_value=UserProfile()):
        authorization_service._storeAuthData(NO_REFRESH_AUTH_RESPONSE)
    mock_callback = Mock()
    mock_reply = Mock()
    mock_reply.error = Mock(return_value=QNetworkReply.NetworkError.NoError)
    http_mock = Mock()
    http_mock.get = lambda url, headers_dict, callback, error_callback: callback(mock_reply)
    http_mock.readJSON = Mock(return_value={'data': {'user_id': 'id_ego_or_superego', 'username': 'Ghostkeeper'}})
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance', MagicMock(return_value=http_mock)):
        authorization_service._parseJWT(mock_callback)
    mock_callback.assert_called_once()
    profile_reply = mock_callback.call_args_list[0][0][0]
    assert profile_reply.user_id == 'id_ego_or_superego'
    assert profile_reply.username == 'Ghostkeeper'

def test__parseJWTFailOnRefresh():
    if False:
        return 10
    '\n    Tries to refresh the authentication token using an invalid refresh token. The request should fail.\n    '
    authorization_service = AuthorizationService(OAUTH_SETTINGS, Preferences())
    with patch.object(AuthorizationService, 'getUserProfile', return_value=UserProfile()):
        authorization_service._storeAuthData(SUCCESSFUL_AUTH_RESPONSE)
    mock_callback = Mock()
    mock_reply = Mock()
    mock_reply.error = Mock(return_value=QNetworkReply.NetworkError.AuthenticationRequiredError)
    http_mock = Mock()
    http_mock.get = lambda url, headers_dict, callback, error_callback: callback(mock_reply)
    http_mock.post = lambda url, data, headers_dict, callback, error_callback: callback(mock_reply)
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.readJSON', Mock(return_value={'error_description': 'Mock a failed request!'})):
        with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance', MagicMock(return_value=http_mock)):
            authorization_service._parseJWT(mock_callback)
    mock_callback.assert_called_once_with(None)

def test__parseJWTSucceedOnRefresh():
    if False:
        print('Hello World!')
    '\n    Tries to refresh the authentication token using a valid refresh token. The request should succeed.\n    '
    authorization_service = AuthorizationService(OAUTH_SETTINGS, Preferences())
    authorization_service.initialize()
    with patch.object(AuthorizationService, 'getUserProfile', return_value=UserProfile()):
        authorization_service._storeAuthData(EXPIRED_AUTH_RESPONSE)
    mock_callback = Mock()
    mock_reply_success = Mock()
    mock_reply_success.error = Mock(return_value=QNetworkReply.NetworkError.NoError)
    mock_reply_failure = Mock()
    mock_reply_failure.error = Mock(return_value=QNetworkReply.NetworkError.AuthenticationRequiredError)
    http_mock = Mock()

    def mock_get(url, headers_dict, callback, error_callback):
        if False:
            while True:
                i = 10
        if headers_dict == {'Authorization': 'Bearer beep'}:
            callback(mock_reply_success)
        else:
            callback(mock_reply_failure)
    http_mock.get = mock_get
    http_mock.readJSON = Mock(return_value={'data': {'user_id': 'user_idea', 'username': 'Ghostkeeper'}})

    def mock_refresh(self, refresh_token, callback):
        if False:
            while True:
                i = 10
        callback(SUCCESSFUL_AUTH_RESPONSE)
    with patch('cura.OAuth2.AuthorizationHelpers.AuthorizationHelpers.getAccessTokenUsingRefreshToken', mock_refresh):
        with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance', MagicMock(return_value=http_mock)):
            authorization_service._parseJWT(mock_callback)
    mock_callback.assert_called_once()
    profile_reply = mock_callback.call_args_list[0][0][0]
    assert profile_reply.user_id == 'user_idea'
    assert profile_reply.username == 'Ghostkeeper'

def test_initialize():
    if False:
        print('Hello World!')
    original_preference = MagicMock()
    initialize_preferences = MagicMock()
    authorization_service = AuthorizationService(OAUTH_SETTINGS, original_preference)
    authorization_service.initialize(initialize_preferences)
    initialize_preferences.addPreference.assert_called_once_with('test/auth_data', '{}')
    original_preference.addPreference.assert_not_called()

def test_refreshAccessTokenFailed():
    if False:
        while True:
            i = 10
    '\n    Test if the authentication is reset once the refresh token fails to refresh access.\n    '
    authorization_service = AuthorizationService(OAUTH_SETTINGS, Preferences())
    authorization_service.initialize()

    def mock_refresh(self, refresh_token, callback):
        if False:
            while True:
                i = 10
        callback(FAILED_AUTH_RESPONSE)
    mock_reply = Mock()
    mock_reply.error = Mock(return_value=QNetworkReply.NetworkError.AuthenticationRequiredError)
    http_mock = Mock()
    http_mock.get = lambda url, headers_dict, callback, error_callback: callback(mock_reply)
    http_mock.post = lambda url, data, headers_dict, callback, error_callback: callback(mock_reply)
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.readJSON', Mock(return_value={'error_description': 'Mock a failed request!'})):
        with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance', MagicMock(return_value=http_mock)):
            authorization_service._storeAuthData(SUCCESSFUL_AUTH_RESPONSE)
            authorization_service.onAuthStateChanged.emit = MagicMock()
            with patch('cura.OAuth2.AuthorizationHelpers.AuthorizationHelpers.getAccessTokenUsingRefreshToken', mock_refresh):
                authorization_service.refreshAccessToken()
                assert authorization_service.onAuthStateChanged.emit.called_with(False)

def test_refreshAccesTokenWithoutData():
    if False:
        i = 10
        return i + 15
    authorization_service = AuthorizationService(OAUTH_SETTINGS, Preferences())
    authorization_service.initialize()
    authorization_service.onAuthStateChanged.emit = MagicMock()
    authorization_service.refreshAccessToken()
    authorization_service.onAuthStateChanged.emit.assert_not_called()

def test_failedLogin() -> None:
    if False:
        i = 10
        return i + 15
    authorization_service = AuthorizationService(OAUTH_SETTINGS, Preferences())
    authorization_service.onAuthenticationError.emit = MagicMock()
    authorization_service.onAuthStateChanged.emit = MagicMock()
    authorization_service.initialize()
    authorization_service._onAuthStateChanged(FAILED_AUTH_RESPONSE)
    assert authorization_service.onAuthenticationError.emit.call_count == 1
    assert authorization_service.onAuthStateChanged.emit.call_count == 0
    assert authorization_service.getUserProfile() is None
    assert authorization_service.getAccessToken() is None

@patch.object(AuthorizationService, 'getUserProfile')
def test_storeAuthData(get_user_profile) -> None:
    if False:
        i = 10
        return i + 15
    preferences = Preferences()
    authorization_service = AuthorizationService(OAUTH_SETTINGS, preferences)
    authorization_service.initialize()
    authorization_service._storeAuthData(SUCCESSFUL_AUTH_RESPONSE)
    preference_value = preferences.getValue(OAUTH_SETTINGS.AUTH_DATA_PREFERENCE_KEY)
    assert preference_value is not None and preference_value != {}
    second_auth_service = AuthorizationService(OAUTH_SETTINGS, preferences)
    second_auth_service.initialize()
    second_auth_service.loadAuthDataFromPreferences()
    assert second_auth_service.getAccessToken() == SUCCESSFUL_AUTH_RESPONSE.access_token

@patch.object(LocalAuthorizationServer, 'stop')
@patch.object(LocalAuthorizationServer, 'start')
@patch.object(QDesktopServices, 'openUrl')
def test_localAuthServer(QDesktopServices_openUrl, start_auth_server, stop_auth_server) -> None:
    if False:
        return 10
    preferences = Preferences()
    authorization_service = AuthorizationService(OAUTH_SETTINGS, preferences)
    authorization_service.startAuthorizationFlow()
    assert QDesktopServices_openUrl.call_count == 1
    assert start_auth_server.call_count == 1
    assert stop_auth_server.call_count == 0
    authorization_service._onAuthStateChanged(FAILED_AUTH_RESPONSE)
    assert stop_auth_server.call_count == 1

def test_loginAndLogout() -> None:
    if False:
        i = 10
        return i + 15
    preferences = Preferences()
    authorization_service = AuthorizationService(OAUTH_SETTINGS, preferences)
    authorization_service.onAuthenticationError.emit = MagicMock()
    authorization_service.onAuthStateChanged.emit = MagicMock()
    authorization_service.initialize()
    mock_reply = Mock()
    mock_reply.error = Mock(return_value=QNetworkReply.NetworkError.NoError)
    http_mock = Mock()
    http_mock.get = lambda url, headers_dict, callback, error_callback: callback(mock_reply)
    http_mock.readJSON = Mock(return_value={'data': {'user_id': 'di_resu', 'username': 'Emanresu'}})
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance', MagicMock(return_value=http_mock)):
        authorization_service._onAuthStateChanged(SUCCESSFUL_AUTH_RESPONSE)
    assert authorization_service.onAuthenticationError.emit.call_count == 0
    assert authorization_service.onAuthStateChanged.emit.call_count == 1
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance', MagicMock(return_value=http_mock)):

        def callback(profile):
            if False:
                for i in range(10):
                    print('nop')
            assert profile is not None
        authorization_service.getUserProfile(callback)
    assert authorization_service.getAccessToken() == 'beep'
    assert preferences.getValue('test/auth_data') is not None
    authorization_service.deleteAuthData()
    assert authorization_service.onAuthStateChanged.emit.call_count == 2
    with patch('UM.TaskManagement.HttpRequestManager.HttpRequestManager.getInstance', MagicMock(return_value=http_mock)):

        def callback(profile):
            if False:
                while True:
                    i = 10
            assert profile is None
        authorization_service.getUserProfile(callback)
    assert preferences.getValue('test/auth_data') == '{}'

def test_wrongServerResponses() -> None:
    if False:
        print('Hello World!')
    authorization_service = AuthorizationService(OAUTH_SETTINGS, Preferences())
    authorization_service.initialize()
    authorization_service._onAuthStateChanged(MALFORMED_AUTH_RESPONSE)

    def callback(profile):
        if False:
            while True:
                i = 10
        assert profile is None
    authorization_service.getUserProfile(callback)

def test__generate_auth_url() -> None:
    if False:
        print('Hello World!')
    preferences = Preferences()
    authorization_service = AuthorizationService(OAUTH_SETTINGS, preferences)
    query_parameters_dict = {'client_id': '', 'redirect_uri': OAUTH_SETTINGS.CALLBACK_URL, 'scope': OAUTH_SETTINGS.CLIENT_SCOPES, 'response_type': 'code'}
    auth_url = authorization_service._generate_auth_url(query_parameters_dict, force_browser_logout=False)
    assert MYCLOUD_LOGOFF_URL + '&next=' not in auth_url
    auth_url = authorization_service._generate_auth_url(query_parameters_dict, force_browser_logout=True)
    assert MYCLOUD_LOGOFF_URL + '&next=' in auth_url
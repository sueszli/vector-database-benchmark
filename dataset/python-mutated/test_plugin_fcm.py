import os
import sys
from unittest import mock
import pytest
import requests
import json
from apprise import Apprise
from apprise.plugins.NotifyFCM import NotifyFCM
from helpers import AppriseURLTester
try:
    from apprise.plugins.NotifyFCM.oauth import GoogleOAuth
    from apprise.plugins.NotifyFCM.common import FCM_MODES
    from apprise.plugins.NotifyFCM.priority import FCMPriorityManager, FCM_PRIORITIES
    from apprise.plugins.NotifyFCM.color import FCMColorManager
    from cryptography.exceptions import UnsupportedAlgorithm
except ImportError:
    pass
import logging
logging.disable(logging.CRITICAL)
PRIVATE_KEYFILE_DIR = os.path.join(os.path.dirname(__file__), 'var', 'fcm')
FCM_KEYFILE = os.path.join(PRIVATE_KEYFILE_DIR, 'service_account.json')
apprise_url_tests = (('fcm://', {'instance': TypeError}), ('fcm://:@/', {'instance': TypeError}), ('fcm://project@%20%20/', {'instance': TypeError}), ('fcm://apikey/', {'instance': NotifyFCM, 'notify_response': False}), ('fcm://apikey/device', {'instance': NotifyFCM, 'privacy_url': 'fcm://a...y/device'}), ('fcm://apikey/#topic', {'instance': NotifyFCM, 'privacy_url': 'fcm://a...y/%23topic'}), ('fcm://apikey/device?mode=invalid', {'instance': TypeError}), ('fcm://apikey/#topic1/device/%20/', {'instance': NotifyFCM}), ('fcm://apikey?to=#topic1,device', {'instance': NotifyFCM}), ('fcm://?apikey=abc123&to=device', {'instance': NotifyFCM}), ('fcm://?apikey=abc123&to=device&image=yes', {'instance': NotifyFCM}), ('fcm://?apikey=abc123&to=device&color=no', {'instance': NotifyFCM}), ('fcm://?apikey=abc123&to=device&color=aabbcc', {'instance': NotifyFCM}), ('fcm://?apikey=abc123&to=device&image_url=http://example.com/interesting.jpg', {'instance': NotifyFCM}), ('fcm://?apikey=abc123&to=device&image_url=http://example.com/interesting.jpg&image=no', {'instance': NotifyFCM}), ('fcm://?apikey=abc123&to=device&+key=value&+key2=value2', {'instance': NotifyFCM}), ('fcm://%20?to=device&keyfile=/invalid/path', {'instance': TypeError}), ('fcm://project_id?to=device&keyfile=/invalid/path', {'instance': NotifyFCM, 'response': False}), ('fcm://?to=device&project=project_id&keyfile=/invalid/path', {'instance': NotifyFCM, 'response': False}), ('fcm://project_id?to=device&mode=oauth2', {'instance': TypeError}), ('fcm://project_id?to=device&mode=oauth2&keyfile=/invalid/path', {'instance': NotifyFCM, 'response': False}), ('fcm://apikey/#topic1/device/?mode=legacy', {'instance': NotifyFCM, 'response': False, 'requests_response_code': 999}), ('fcm://apikey/#topic1/device/?mode=legacy', {'instance': NotifyFCM, 'test_requests_exceptions': True}), ('fcm://project/#topic1/device/?mode=oauth2&keyfile=file://{}'.format(os.path.join(os.path.dirname(__file__), 'var', 'fcm', 'service_account.json')), {'instance': NotifyFCM, 'response': False, 'requests_response_code': 999}), ('fcm://projectid/#topic1/device/?mode=oauth2&keyfile=file://{}'.format(os.path.join(os.path.dirname(__file__), 'var', 'fcm', 'service_account.json')), {'instance': NotifyFCM, 'test_requests_exceptions': True}))

@pytest.fixture
def mock_post(mocker):
    if False:
        return 10
    '\n    Prepare a good OAuth mock response.\n    '
    mock_thing = mocker.patch('requests.post')
    response = mock.Mock()
    response.content = json.dumps({'access_token': 'ya29.c.abcd', 'expires_in': 3599, 'token_type': 'Bearer'})
    response.status_code = requests.codes.ok
    mock_thing.return_value = response
    return mock_thing

@pytest.fixture
def mock_post_legacy(mocker):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prepare a good legacy mock response.\n    '
    mock_thing = mocker.patch('requests.post')
    response = mock.Mock()
    response.status_code = requests.codes.ok
    mock_thing.return_value = response
    return mock_thing

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyFCM() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_legacy_default(mock_post_legacy):
    if False:
        while True:
            i = 10
    '\n    NotifyFCM() Legacy/APIKey default checks.\n    '
    obj = Apprise.instantiate('fcm://abc123/device/?+key=value&+key2=value2&image_url=https://example.com/interesting.png')
    assert obj.notify('test') is True
    assert mock_post_legacy.call_count == 1
    assert mock_post_legacy.call_args_list[0][0][0] == 'https://fcm.googleapis.com/fcm/send'
    payload = mock_post_legacy.mock_calls[0][2]
    data = json.loads(payload['data'])
    assert 'data' in data
    assert isinstance(data, dict)
    assert 'key' in data['data']
    assert data['data']['key'] == 'value'
    assert 'key2' in data['data']
    assert data['data']['key2'] == 'value2'
    assert 'notification' in data
    assert isinstance(data['notification'], dict)
    assert 'notification' in data['notification']
    assert isinstance(data['notification']['notification'], dict)
    assert 'image' in data['notification']['notification']
    assert data['notification']['notification']['image'] == 'https://example.com/interesting.png'

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_legacy_priorities(mock_post_legacy):
    if False:
        print('Hello World!')
    '\n    NotifyFCM() Legacy/APIKey priorities checks.\n    '
    obj = Apprise.instantiate('fcm://abc123/device/?priority=low')
    assert mock_post_legacy.call_count == 0
    assert obj.notify(title='title', body='body') is True
    assert mock_post_legacy.call_count == 1
    assert mock_post_legacy.call_args_list[0][0][0] == 'https://fcm.googleapis.com/fcm/send'
    payload = mock_post_legacy.mock_calls[0][2]
    data = json.loads(payload['data'])
    assert 'data' not in data
    assert 'notification' in data
    assert isinstance(data['notification'], dict)
    assert 'notification' in data['notification']
    assert isinstance(data['notification']['notification'], dict)
    assert 'image' not in data['notification']['notification']
    assert 'priority' in data
    assert data['priority'] == 'normal'

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_legacy_no_colors(mock_post_legacy):
    if False:
        print('Hello World!')
    '\n    NotifyFCM() Legacy/APIKey `color=no` checks.\n    '
    obj = Apprise.instantiate('fcm://abc123/device/?color=no')
    assert mock_post_legacy.call_count == 0
    assert obj.notify(title='title', body='body') is True
    assert mock_post_legacy.call_count == 1
    assert mock_post_legacy.call_args_list[0][0][0] == 'https://fcm.googleapis.com/fcm/send'
    payload = mock_post_legacy.mock_calls[0][2]
    data = json.loads(payload['data'])
    assert 'data' not in data
    assert 'notification' in data
    assert isinstance(data['notification'], dict)
    assert 'notification' in data['notification']
    assert isinstance(data['notification']['notification'], dict)
    assert 'image' not in data['notification']['notification']
    assert 'color' not in data['notification']['notification']

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_legacy_colors(mock_post_legacy):
    if False:
        i = 10
        return i + 15
    '\n    NotifyFCM() Legacy/APIKey colors checks.\n    '
    obj = Apprise.instantiate('fcm://abc123/device/?color=AA001b')
    assert mock_post_legacy.call_count == 0
    assert obj.notify(title='title', body='body') is True
    assert mock_post_legacy.call_count == 1
    assert mock_post_legacy.call_args_list[0][0][0] == 'https://fcm.googleapis.com/fcm/send'
    payload = mock_post_legacy.mock_calls[0][2]
    data = json.loads(payload['data'])
    assert 'data' not in data
    assert 'notification' in data
    assert isinstance(data['notification'], dict)
    assert 'notification' in data['notification']
    assert isinstance(data['notification']['notification'], dict)
    assert 'image' not in data['notification']['notification']
    assert 'color' in data['notification']['notification']
    assert data['notification']['notification']['color'] == '#aa001b'

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_oauth_default(mock_post):
    if False:
        return 10
    '\n    NotifyFCM() general OAuth checks - success.\n    Test using a valid Project ID and key file.\n    '
    obj = Apprise.instantiate(f'fcm://mock-project-id/device/#topic/?keyfile={FCM_KEYFILE}')
    assert obj.notify('test') is True
    assert mock_post.call_count == 3
    assert mock_post.call_args_list[0][0][0] == 'https://accounts.google.com/o/oauth2/token'
    assert mock_post.call_args_list[1][0][0] == 'https://fcm.googleapis.com/v1/projects/mock-project-id/messages:send'
    assert mock_post.call_args_list[2][0][0] == 'https://fcm.googleapis.com/v1/projects/mock-project-id/messages:send'

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_oauth_invalid_project_id(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyFCM() OAuth checks, with invalid project id.\n    '
    obj = Apprise.instantiate(f'fcm://invalid_project_id/device/?keyfile={FCM_KEYFILE}')
    assert obj.notify('test') is False
    assert mock_post.call_count == 0

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_oauth_keyfile_error(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifyFCM() OAuth checks, while unable to read key file.\n    '
    obj = Apprise.instantiate(f'fcm://mock-project-id/device/?keyfile={FCM_KEYFILE}')
    with mock.patch('builtins.open', side_effect=OSError):
        assert obj.notify('test') is False
    assert mock_post.call_count == 0

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_oauth_data_parameters(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyFCM() OAuth checks, success.\n    Test using a valid Project ID and data parameters.\n    '
    obj = Apprise.instantiate(f'fcm://mock-project-id/device/#topic/?keyfile={FCM_KEYFILE}&+key=value&+key2=value2&image_url=https://example.com/interesting.png')
    assert mock_post.call_count == 0
    assert obj.notify('test') is True
    assert mock_post.call_count == 3
    assert mock_post.call_args_list[0][0][0] == 'https://accounts.google.com/o/oauth2/token'
    assert mock_post.call_args_list[1][0][0] == 'https://fcm.googleapis.com/v1/projects/mock-project-id/messages:send'
    payload = mock_post.mock_calls[1][2]
    data = json.loads(payload['data'])
    assert 'message' in data
    assert isinstance(data['message'], dict)
    assert 'data' in data['message']
    assert isinstance(data['message']['data'], dict)
    assert 'key' in data['message']['data']
    assert data['message']['data']['key'] == 'value'
    assert 'key2' in data['message']['data']
    assert data['message']['data']['key2'] == 'value2'
    assert 'notification' in data['message']
    assert isinstance(data['message']['notification'], dict)
    assert 'image' in data['message']['notification']
    assert data['message']['notification']['image'] == 'https://example.com/interesting.png'
    assert mock_post.call_args_list[2][0][0] == 'https://fcm.googleapis.com/v1/projects/mock-project-id/messages:send'
    payload = mock_post.mock_calls[2][2]
    data = json.loads(payload['data'])
    assert 'message' in data
    assert isinstance(data['message'], dict)
    assert 'data' in data['message']
    assert isinstance(data['message']['data'], dict)
    assert 'key' in data['message']['data']
    assert data['message']['data']['key'] == 'value'
    assert 'key2' in data['message']['data']
    assert data['message']['data']['key2'] == 'value2'
    assert 'notification' in data['message']
    assert isinstance(data['message']['notification'], dict)
    assert 'image' in data['message']['notification']
    assert data['message']['notification']['image'] == 'https://example.com/interesting.png'

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_oauth_priorities(mock_post):
    if False:
        print('Hello World!')
    '\n    Verify priorities work as intended.\n    '
    obj = Apprise.instantiate(f'fcm://mock-project-id/device/?keyfile={FCM_KEYFILE}&priority=high')
    assert mock_post.call_count == 0
    assert obj.notify(title='title', body='body') is True
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://accounts.google.com/o/oauth2/token'
    assert mock_post.call_args_list[1][0][0] == 'https://fcm.googleapis.com/v1/projects/mock-project-id/messages:send'
    payload = mock_post.mock_calls[1][2]
    data = json.loads(payload['data'])
    assert 'message' in data
    assert isinstance(data['message'], dict)
    assert 'data' not in data['message']
    assert 'notification' in data['message']
    assert isinstance(data['message']['notification'], dict)
    assert 'image' not in data['message']['notification']
    assert data['message']['apns']['headers']['apns-priority'] == '10'
    assert data['message']['webpush']['headers']['Urgency'] == 'high'
    assert data['message']['android']['priority'] == 'HIGH'

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_oauth_no_colors(mock_post):
    if False:
        print('Hello World!')
    '\n    Verify `color=no` work as intended.\n    '
    obj = Apprise.instantiate(f'fcm://mock-project-id/device/?keyfile={FCM_KEYFILE}&color=no')
    assert mock_post.call_count == 0
    assert obj.notify(title='title', body='body') is True
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://accounts.google.com/o/oauth2/token'
    assert mock_post.call_args_list[1][0][0] == 'https://fcm.googleapis.com/v1/projects/mock-project-id/messages:send'
    payload = mock_post.mock_calls[1][2]
    data = json.loads(payload['data'])
    assert 'message' in data
    assert isinstance(data['message'], dict)
    assert 'data' not in data['message']
    assert 'notification' in data['message']
    assert isinstance(data['message']['notification'], dict)
    assert 'color' not in data['message']['notification']

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_oauth_colors(mock_post):
    if False:
        return 10
    '\n    Verify colors work as intended.\n    '
    obj = Apprise.instantiate(f'fcm://mock-project-id/device/?keyfile={FCM_KEYFILE}&color=#12AAbb')
    assert mock_post.call_count == 0
    assert obj.notify(title='title', body='body') is True
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'https://accounts.google.com/o/oauth2/token'
    assert mock_post.call_args_list[1][0][0] == 'https://fcm.googleapis.com/v1/projects/mock-project-id/messages:send'
    payload = mock_post.mock_calls[1][2]
    data = json.loads(payload['data'])
    assert 'message' in data
    assert isinstance(data['message'], dict)
    assert 'data' not in data['message']
    assert 'notification' in data['message']
    assert isinstance(data['message']['notification'], dict)
    assert 'color' in data['message']['android']['notification']
    assert data['message']['android']['notification']['color'] == '#12aabb'

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_keyfile_parse_default(mock_post):
    if False:
        i = 10
        return i + 15
    '\n    NotifyFCM() KeyFile Tests\n    '
    oauth = GoogleOAuth()
    assert oauth.access_token is None
    assert oauth.load(FCM_KEYFILE) is True
    assert oauth.access_token is not None
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://accounts.google.com/o/oauth2/token'
    mock_post.reset_mock()
    assert oauth.access_token is not None
    assert mock_post.call_count == 0

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_keyfile_parse_no_expiry(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test case without `expires_in` entry.\n    '
    mock_post.return_value.content = json.dumps({'access_token': 'ya29.c.abcd', 'token_type': 'Bearer'})
    oauth = GoogleOAuth()
    assert oauth.load(FCM_KEYFILE) is True
    assert oauth.access_token is not None
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://accounts.google.com/o/oauth2/token'

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_keyfile_parse_user_agent(mock_post):
    if False:
        print('Hello World!')
    '\n    Test case with `user-agent` override.\n    '
    oauth = GoogleOAuth(user_agent='test-agent-override')
    assert oauth.load(FCM_KEYFILE) is True
    assert oauth.access_token is not None
    assert mock_post.call_count == 1
    assert mock_post.call_args_list[0][0][0] == 'https://accounts.google.com/o/oauth2/token'

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_keyfile_parse_keyfile_failures(mock_post: mock.Mock):
    if False:
        while True:
            i = 10
    '\n    Test some errors that can get thrown when trying to handle\n    the `service_account.json` file.\n    '
    oauth = GoogleOAuth()
    with mock.patch('builtins.open', side_effect=OSError):
        assert oauth.load(FCM_KEYFILE) is False
        assert oauth.access_token is None
    oauth = GoogleOAuth()
    with mock.patch('json.loads', side_effect=([],)):
        assert oauth.load(FCM_KEYFILE) is False
        assert oauth.access_token is None
    oauth = GoogleOAuth()
    with mock.patch('cryptography.hazmat.primitives.serialization.load_pem_private_key', side_effect=ValueError('')):
        assert oauth.load(FCM_KEYFILE) is False
        assert oauth.access_token is None
    oauth = GoogleOAuth()
    with mock.patch('cryptography.hazmat.primitives.serialization.load_pem_private_key', side_effect=TypeError('')):
        assert oauth.load(FCM_KEYFILE) is False
        assert oauth.access_token is None
    oauth = GoogleOAuth()
    with mock.patch('cryptography.hazmat.primitives.serialization.load_pem_private_key', side_effect=UnsupportedAlgorithm('')):
        assert oauth.load(FCM_KEYFILE) is False
        assert oauth.access_token is None
    assert mock_post.mock_calls == []

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_keyfile_parse_token_failures(mock_post):
    if False:
        while True:
            i = 10
    '\n    Test some web errors that can occur when speaking upstream\n    with Google to get our token generated.\n    '
    mock_post.return_value.status_code = requests.codes.internal_server_error
    oauth = GoogleOAuth()
    assert oauth.load(FCM_KEYFILE) is True
    assert oauth.access_token is None
    mock_post.return_value.status_code = requests.codes.ok
    bad_response_1 = mock.Mock()
    bad_response_1.content = json.dumps({'expires_in': 3599, 'token_type': 'Bearer'})
    bad_response_2 = mock.Mock()
    bad_response_2.content = '{'
    mock_post.return_value = None
    for side_effect in (requests.RequestException(), bad_response_1, bad_response_2):
        mock_post.side_effect = side_effect
        oauth = GoogleOAuth()
        assert oauth.load(FCM_KEYFILE) is True
        assert oauth.access_token is None

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_bad_keyfile_parse():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyFCM() KeyFile Bad Service Account Type Tests\n    '
    path = os.path.join(PRIVATE_KEYFILE_DIR, 'service_account-bad-type.json')
    oauth = GoogleOAuth()
    assert oauth.load(path) is False

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_keyfile_missing_entries_parse(tmpdir):
    if False:
        return 10
    '\n    NotifyFCM() KeyFile Missing Entries Test\n    '
    path = os.path.join(PRIVATE_KEYFILE_DIR, 'service_account.json')
    with open(path, mode='r', encoding='utf-8') as fp:
        content = json.loads(fp.read())
    path = tmpdir.join('fcm_keyfile.json')
    for entry in ('client_email', 'private_key_id', 'private_key', 'type', 'project_id'):
        assert entry in content
        content_copy = content.copy()
        del content_copy[entry]
        assert entry not in content_copy
        path.write(json.dumps(content_copy))
        oauth = GoogleOAuth()
        assert oauth.load(str(path)) is False
    path.write('{')
    oauth = GoogleOAuth()
    assert oauth.load(str(path)) is False

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_priority_manager():
    if False:
        i = 10
        return i + 15
    '\n    NotifyFCM() FCMPriorityManager() Testing\n    '
    for mode in FCM_MODES:
        for priority in FCM_PRIORITIES:
            instance = FCMPriorityManager(mode, priority)
            assert isinstance(instance.payload(), dict)
            assert bool(instance)
            assert instance.payload()
            assert str(instance) == priority
    instance = FCMPriorityManager(mode)
    assert isinstance(instance.payload(), dict)
    assert not bool(instance)
    assert not instance.payload()
    assert str(instance) == ''
    with pytest.raises(TypeError):
        instance = FCMPriorityManager(mode, 'invalid')
    with pytest.raises(TypeError):
        instance = FCMPriorityManager('invald', 'high')

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
def test_plugin_fcm_color_manager():
    if False:
        return 10
    '\n    NotifyFCM() FCMColorManager() Testing\n    '
    instance = FCMColorManager('no')
    assert bool(instance) is False
    assert instance.get() is None
    assert str(instance) == 'no'
    instance = FCMColorManager('yes')
    assert isinstance(instance.get(), str)
    assert len(instance.get()) == 7
    assert instance.get()[0] == '#'
    assert str(instance) == 'yes'
    assert bool(instance) is True
    instance = FCMColorManager('#A2B3A4')
    assert isinstance(instance.get(), str)
    assert instance.get() == '#a2b3a4'
    assert bool(instance) is True
    assert str(instance) == 'a2b3a4'
    instance = FCMColorManager('A2B3A4')
    assert isinstance(instance.get(), str)
    assert instance.get() == '#a2b3a4'
    assert bool(instance) is True
    assert str(instance) == 'a2b3a4'
    instance = FCMColorManager('AC4')
    assert isinstance(instance.get(), str)
    assert instance.get() == '#aacc44'
    assert bool(instance) is True
    assert str(instance) == 'aacc44'

@pytest.mark.skipif('cryptography' in sys.modules, reason='Requires that cryptography NOT be installed')
def test_plugin_fcm_cryptography_import_error():
    if False:
        print('Hello World!')
    '\n    NotifyFCM Cryptography loading failure\n    '
    path = os.path.join(PRIVATE_KEYFILE_DIR, 'service_account.json')
    obj = Apprise.instantiate('fcm://mock-project-id/device/#topic/?keyfile={}'.format(str(path)))
    assert obj is None

@pytest.mark.skipif('cryptography' not in sys.modules, reason='Requires cryptography')
@mock.patch('requests.post')
def test_plugin_fcm_edge_cases(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyFCM() Edge Cases\n\n    '
    response = mock.Mock()
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    obj = NotifyFCM('project', 'api:123', targets='device')
    assert obj is not None
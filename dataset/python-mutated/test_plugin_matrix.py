from unittest import mock
import os
import requests
import pytest
from apprise import Apprise, AppriseAsset, AppriseAttachment, NotifyType
from json import dumps
from apprise.plugins.NotifyMatrix import NotifyMatrix
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
MATRIX_GOOD_RESPONSE = dumps({'room_id': '!abc123:localhost', 'room_alias': '#abc123:localhost', 'joined_rooms': ['!abc123:localhost', '!def456:localhost'], 'access_token': 'abcd1234', 'home_server': 'localhost'})
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('matrix://', {'instance': None}), ('matrixs://', {'instance': None}), ('matrix://localhost?mode=off', {'instance': NotifyMatrix, 'response': False}), ('matrix://localhost', {'instance': TypeError}), ('matrix://user:pass@localhost/#room1/#room2/#room3', {'instance': NotifyMatrix, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('matrix://user:pass@localhost/#room1/#room2/!room1', {'instance': NotifyMatrix, 'response': False, 'requests_response_code': 999}), ('matrix://user:pass@localhost:1234/#room', {'instance': NotifyMatrix, 'test_requests_exceptions': True, 'privacy_url': 'matrix://user:****@localhost:1234/'}), ('matrix://user:token@localhost?mode=matrix&format=text', {'instance': NotifyMatrix, 'response': False}), ('matrix://user:token@localhost?mode=matrix&format=html', {'instance': NotifyMatrix}), ('matrix://user:token@localhost:123/#general/?version=3', {'instance': NotifyMatrix, 'requests_response_text': MATRIX_GOOD_RESPONSE, 'privacy_url': 'matrix://user:****@localhost:123'}), ('matrixs://user:token@localhost/#general?v=2', {'instance': NotifyMatrix, 'requests_response_text': MATRIX_GOOD_RESPONSE, 'privacy_url': 'matrixs://user:****@localhost'}), ('matrix://user:token@localhost:123/#general/?v=invalid', {'instance': TypeError}), ('matrix://user:token@localhost?mode=slack&format=text', {'instance': NotifyMatrix}), ('matrixs://user:token@localhost?mode=SLACK&format=markdown', {'instance': NotifyMatrix}), ('matrix://user@localhost?mode=SLACK&format=markdown&token=mytoken', {'instance': NotifyMatrix}), ('matrix://_?mode=t2bot&token={}'.format('b' * 64), {'instance': NotifyMatrix, 'privacy_url': 'matrix://b...b/'}), ('matrixs://user:token@localhost?mode=slack&format=markdown&image=True', {'instance': NotifyMatrix}), ('matrixs://user:token@localhost?mode=slack&format=markdown&image=False', {'instance': NotifyMatrix}), ('matrixs://user:pass@hostname:port/#room_alias', {'instance': TypeError}), ('matrixs://user:pass@hostname:0/#room_alias', {'instance': TypeError}), ('matrixs://user:pass@hostname:65536/#room_alias', {'instance': TypeError}), ('matrixs://user@{}?mode=t2bot&format=markdown&image=True'.format('a' * 64), {'instance': NotifyMatrix}), ('matrix://user@{}?mode=t2bot&format=html&image=False'.format('z' * 64), {'instance': NotifyMatrix}), ('matrixs://{}'.format('c' * 64), {'instance': NotifyMatrix, 'test_requests_exceptions': True}), ('https://webhooks.t2bot.io/api/v1/matrix/hook/{}/'.format('d' * 64), {'instance': NotifyMatrix}), ('matrix://user:token@localhost?mode=On', {'instance': TypeError}), ('matrix://token@localhost/?mode=Matrix', {'instance': NotifyMatrix, 'response': False, 'requests_response_code': requests.codes.internal_server_error}), ('matrix://user:token@localhost/mode=matrix', {'instance': NotifyMatrix, 'response': False, 'requests_response_code': 999}), ('matrix://token@localhost:8080/?mode=slack', {'instance': NotifyMatrix, 'test_requests_exceptions': True}), ('matrix://{}/?mode=t2bot'.format('b' * 64), {'instance': NotifyMatrix, 'test_requests_exceptions': True}))

def test_plugin_matrix_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyMatrix() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.put')
@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_matrix_general(mock_post, mock_get, mock_put):
    if False:
        return 10
    '\n    NotifyMatrix() General Tests\n\n    '
    response_obj = {'room_id': '!abc123:localhost', 'room_alias': '#abc123:localhost', 'joined_rooms': ['!abc123:localhost', '!def456:localhost'], 'access_token': 'abcd1234', 'home_server': 'localhost'}
    request = mock.Mock()
    request.content = dumps(response_obj)
    request.status_code = requests.codes.ok
    mock_get.return_value = request
    mock_post.return_value = request
    mock_put.return_value = request
    obj = NotifyMatrix(host='host', targets='#abcd')
    assert isinstance(obj, NotifyMatrix) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body='test') is True
    obj = NotifyMatrix(host='host', user='user', targets='#abcd')
    assert isinstance(obj, NotifyMatrix) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body='test') is True
    obj = NotifyMatrix(host='host', password='passwd', targets='#abcd')
    assert isinstance(obj, NotifyMatrix) is True
    assert isinstance(obj.url(), str) is True
    assert obj.send(body='test') is True
    obj = NotifyMatrix(host='host', user='user', password='passwd', targets='#abcd')
    assert isinstance(obj.url(), str) is True
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.send(body='test') is True
    kwargs = NotifyMatrix.parse_url('matrix://user:passwd@hostname/#abcd?format=html')
    obj = NotifyMatrix(**kwargs)
    assert isinstance(obj.url(), str) is True
    assert isinstance(obj, NotifyMatrix) is True
    obj.send(body='test') is True
    obj.send(title='title', body='test') is True
    kwargs = NotifyMatrix.parse_url('matrix://user:passwd@hostname/#abcd/#abcd:localhost?format=markdown')
    obj = NotifyMatrix(**kwargs)
    assert isinstance(obj.url(), str) is True
    assert isinstance(obj, NotifyMatrix) is True
    obj.send(body='test') is True
    obj.send(title='title', body='test') is True
    kwargs = NotifyMatrix.parse_url('matrix://user:passwd@hostname/#abcd/!abcd:localhost?format=text')
    obj = NotifyMatrix(**kwargs)
    assert isinstance(obj.url(), str) is True
    assert isinstance(obj, NotifyMatrix) is True
    obj.send(body='test') is True
    obj.send(title='title', body='test') is True
    kwargs = NotifyMatrix.parse_url('matrix://user:passwd@hostname/#abcd?msgtype=notice')
    obj = NotifyMatrix(**kwargs)
    assert isinstance(obj.url(), str) is True
    assert isinstance(obj, NotifyMatrix) is True
    obj.send(body='test') is True
    obj.send(title='title', body='test') is True
    with pytest.raises(TypeError):
        kwargs = NotifyMatrix.parse_url('matrix://user:passwd@hostname/#abcd?msgtype=invalid')
        obj = NotifyMatrix(**kwargs)
    ro = response_obj.copy()
    del ro['access_token']
    request.content = dumps(ro)
    request.status_code = 404
    assert obj.send(body='test') is False
    obj = NotifyMatrix(host='host', user='test', targets='#abcd')
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.send(user='test', password='passwd', body='test') is False
    obj = NotifyMatrix(host='host', user='test', password='passwd', targets='#abcd')
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.send(body='test') is False
    obj = NotifyMatrix(host='host', password='passwd', targets='#abcd')
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.send(body='test') is False
    ro = response_obj.copy()
    ro['joined_rooms'] = []
    request.content = dumps(ro)
    assert obj.send(user='test', password='passwd', body='test') is False
    request.content = dumps(response_obj)
    request.status_code = requests.codes.ok
    response_obj['user_id'] = '@apprise:localhost'
    ro = response_obj.copy()
    del ro['room_id']
    request.content = dumps(ro)
    assert obj.send(user='test', password='passwd', body='test') is False
    request.content = dumps(response_obj)
    request.status_code = requests.codes.ok
    obj = NotifyMatrix(host='host', targets=None)
    assert isinstance(obj, NotifyMatrix) is True
    ro = response_obj.copy()
    ro['joined_rooms'] = []
    request.content = dumps(ro)
    assert obj.send(user='test', password='passwd', body='test') is False
    request.content = dumps(response_obj)
    request.status_code = requests.codes.ok
    assert obj.send(user='test', password='passwd', body='test') is True

@mock.patch('requests.put')
@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_matrix_fetch(mock_post, mock_get, mock_put):
    if False:
        i = 10
        return i + 15
    '\n    NotifyMatrix() Server Fetch/API Tests\n\n    '
    response_obj = {'room_id': '!abc123:localhost', 'room_alias': '#abc123:localhost', 'joined_rooms': ['!abc123:localhost', '!def456:localhost'], 'access_token': 'abcd1234', 'user_id': '@apprise:localhost', 'home_server': 'localhost'}

    def fetch_failed(url, *args, **kwargs):
        if False:
            while True:
                i = 10
        request = mock.Mock()
        request.status_code = requests.codes.ok
        request.content = dumps(response_obj)
        if url.find('/rooms/') > -1:
            request.status_code = 403
            request.content = dumps({u'errcode': u'M_UNKNOWN', u'error': u'Internal server error'})
        return request
    mock_put.side_effect = fetch_failed
    mock_get.side_effect = fetch_failed
    mock_post.side_effect = fetch_failed
    obj = NotifyMatrix(host='host', user='user', password='passwd', include_image=True)
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.send(user='test', password='passwd', body='test') is False
    asset = AppriseAsset(image_path_mask=False, image_url_mask=False)
    obj = NotifyMatrix(host='host', user='user', password='passwd', asset=asset)
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.send(user='test', password='passwd', body='test') is False
    response_obj = {'access_token': 'abcd1234', 'user_id': '@apprise:localhost', 'home_server': 'localhost', 'room_id': '!abc123:localhost'}
    mock_get.side_effect = None
    mock_post.side_effect = None
    mock_put.side_effect = None
    request = mock.Mock()
    request.status_code = requests.codes.ok
    request.content = dumps(response_obj)
    mock_post.return_value = request
    mock_get.return_value = request
    mock_put.return_value = request
    obj = NotifyMatrix(host='host', include_image=True)
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.access_token is None
    assert obj._register() is True
    assert obj.access_token is not None
    request.status_code = 429
    request.content = dumps({'retry_after_ms': 1})
    (code, response) = obj._fetch('/retry/apprise/unit/test')
    assert code is False
    request.content = dumps({'error': {'retry_after_ms': 1}})
    (code, response) = obj._fetch('/retry/apprise/unit/test')
    assert code is False
    request.content = dumps({'error': {}})
    (code, response) = obj._fetch('/retry/apprise/unit/test')
    assert code is False

@mock.patch('requests.put')
@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_matrix_auth(mock_post, mock_get, mock_put):
    if False:
        return 10
    '\n    NotifyMatrix() Server Authentication\n\n    '
    response_obj = {'access_token': 'abcd1234', 'user_id': '@apprise:localhost', 'home_server': 'localhost'}
    request = mock.Mock()
    request.status_code = requests.codes.ok
    request.content = dumps(response_obj)
    mock_post.return_value = request
    mock_get.return_value = request
    mock_put.return_value = request
    obj = NotifyMatrix(host='localhost')
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.access_token is None
    assert obj._logout() is True
    assert obj.access_token is None
    assert obj._register() is True
    assert obj.access_token is not None
    assert obj._login() is True
    assert obj.access_token is not None
    assert obj._logout() is True
    assert obj.access_token is None
    request.status_code = 403
    assert obj._login() is False
    assert obj.access_token is None
    obj.access_token = None
    request.status_code = requests.codes.ok
    ro = response_obj.copy()
    del ro['access_token']
    request.content = dumps(ro)
    assert obj._register() is False
    assert obj.access_token is None
    obj = NotifyMatrix(host='host', user='user', password='password')
    assert isinstance(obj, NotifyMatrix) is True
    assert obj._login() is False
    assert obj.access_token is None
    request.content = '{'
    assert obj._register() is False
    assert obj.access_token is None
    request.status_code = requests.codes.ok
    request.content = dumps(response_obj)
    assert obj._register() is True
    assert obj.access_token is not None
    request.status_code = 403
    assert obj._logout() is False
    assert obj.access_token is not None
    request.status_code = requests.codes.ok
    request.content = dumps(response_obj)
    assert obj._register() is True
    assert obj.access_token is not None
    request.status_code = 403
    request.content = dumps({u'errcode': u'M_UNKNOWN_TOKEN', u'error': u'Access Token unknown or expired'})
    assert obj._logout() is True
    assert obj.access_token is None

@mock.patch('requests.put')
@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_matrix_rooms(mock_post, mock_get, mock_put):
    if False:
        print('Hello World!')
    '\n    NotifyMatrix() Room Testing\n\n    '
    response_obj = {'access_token': 'abcd1234', 'user_id': '@apprise:localhost', 'home_server': 'localhost', 'joined_rooms': ['!abc123:localhost', '!def456:localhost'], 'room_id': '!abc123:localhost'}
    request = mock.Mock()
    request.status_code = requests.codes.ok
    request.content = dumps(response_obj)
    mock_post.return_value = request
    mock_get.return_value = request
    mock_put.return_value = request
    obj = NotifyMatrix(host='host')
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.access_token is None
    assert obj._room_join('#abc123') is None
    assert obj._register() is True
    assert obj.access_token is not None
    assert obj._room_join('!abc123') == response_obj['room_id']
    assert len(obj._room_cache) == 1
    assert obj._room_join('!abc123') == response_obj['room_id']
    obj._room_cache = {}
    assert obj._room_join('!abc123:localhost') == response_obj['room_id']
    assert len(obj._room_cache) == 1
    assert obj._room_join('!abc123:localhost') == response_obj['room_id']
    obj._room_cache = {}
    assert obj._room_join('abc123') == response_obj['room_id']
    assert len(obj._room_cache) == 1
    assert obj._room_join('abc123') == response_obj['room_id']
    obj._room_cache = {}
    assert obj._room_join('abc123:localhost') == response_obj['room_id']
    assert len(obj._room_cache) == 1
    assert obj._room_join('abc123:localhost') == response_obj['room_id']
    obj._room_cache = {}
    assert obj._room_join('#abc123:localhost') == response_obj['room_id']
    assert len(obj._room_cache) == 1
    assert obj._room_join('#abc123:localhost') == response_obj['room_id']
    obj._room_cache = {}
    assert obj._room_join('%') is None
    assert obj._room_join(None) is None
    request.status_code = 403
    obj._room_cache = {}
    assert obj._room_join('!abc123') is None
    obj._room_cache = {}
    assert obj._room_join('!abc123:localhost') is None
    obj._room_cache = {}
    assert obj._room_join('abc123') is None
    obj._room_cache = {}
    assert obj._room_join('abc123:localhost') is None
    obj._room_cache = {}
    assert obj._room_join('#abc123:localhost') is None
    request.status_code = requests.codes.ok
    obj = NotifyMatrix(host='host')
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.access_token is None
    assert obj._room_create('#abc123') is None
    assert obj._register() is True
    assert obj.access_token is not None
    assert obj._room_create('!abc123') is None
    assert obj._room_create('!abc123:localhost') is None
    obj._room_cache = {}
    assert obj._room_create('abc123') == response_obj['room_id']
    obj._room_cache = {}
    assert obj._room_create('abc123:localhost') == response_obj['room_id']
    obj._room_cache = {}
    assert obj._room_create('#abc123:localhost') == response_obj['room_id']
    obj._room_cache = {}
    assert obj._room_create('%') is None
    assert obj._room_create(None) is None
    request.status_code = 403
    obj._room_cache = {}
    assert obj._room_create('abc123') is None
    obj._room_cache = {}
    assert obj._room_create('abc123:localhost') is None
    obj._room_cache = {}
    assert obj._room_create('#abc123:localhost') is None
    request.status_code = 403
    request.content = dumps({u'errcode': u'M_ROOM_IN_USE', u'error': u'Room alias already taken'})
    obj._room_cache = {}
    assert obj._room_create('#abc123:localhost') is None
    request.status_code = requests.codes.ok
    request.content = dumps(response_obj)
    obj = NotifyMatrix(host='localhost')
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.access_token is None
    response = obj._joined_rooms()
    assert isinstance(response, list) is True
    assert len(response) == 0
    assert obj._register() is True
    assert obj.access_token is not None
    response = obj._joined_rooms()
    assert isinstance(response, list) is True
    assert len(response) == len(response_obj['joined_rooms'])
    for r in response:
        assert r in response_obj['joined_rooms']
    request.status_code = 403
    response = obj._joined_rooms()
    assert isinstance(response, list) is True
    assert len(response) == 0
    request.status_code = requests.codes.ok
    obj = NotifyMatrix(host='localhost')
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.access_token is None
    assert obj._room_id('#abc123') is None
    assert obj._register() is True
    assert obj.access_token is not None
    assert obj._room_id('!abc123') is None
    assert obj._room_id('!abc123:localhost') is None
    obj._room_cache = {}
    assert obj._room_id('abc123') == response_obj['room_id']
    obj._room_cache = {}
    assert obj._room_id('abc123:localhost') == response_obj['room_id']
    obj._room_cache = {}
    assert obj._room_id('#abc123:localhost') == response_obj['room_id']
    obj._room_cache = {}
    assert obj._room_id('%') is None
    assert obj._room_id(None) is None
    request.status_code = 403
    obj._room_cache = {}
    assert obj._room_id('#abc123:localhost') is None
    del obj

def test_plugin_matrix_url_parsing():
    if False:
        print('Hello World!')
    '\n    NotifyMatrix() URL Testing\n\n    '
    result = NotifyMatrix.parse_url('matrix://user:token@localhost?to=#room')
    assert isinstance(result, dict) is True
    assert len(result['targets']) == 1
    assert '#room' in result['targets']
    result = NotifyMatrix.parse_url('matrix://user:token@localhost?to=#room1,#room2,#room3')
    assert isinstance(result, dict) is True
    assert len(result['targets']) == 3
    assert '#room1' in result['targets']
    assert '#room2' in result['targets']
    assert '#room3' in result['targets']

@mock.patch('requests.put')
@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_matrix_image_errors(mock_post, mock_get, mock_put):
    if False:
        print('Hello World!')
    '\n    NotifyMatrix() Image Error Handling\n\n    '

    def mock_function_handing(url, data, **kwargs):
        if False:
            print('Hello World!')
        '\n        dummy function for handling image posts (as a failure)\n        '
        response_obj = {'room_id': '!abc123:localhost', 'room_alias': '#abc123:localhost', 'joined_rooms': ['!abc123:localhost', '!def456:localhost'], 'access_token': 'abcd1234', 'home_server': 'localhost'}
        request = mock.Mock()
        request.content = dumps(response_obj)
        request.status_code = requests.codes.ok
        if 'm.image' in data:
            request.status_code = 400
        return request
    mock_get.side_effect = mock_function_handing
    mock_post.side_effect = mock_function_handing
    mock_put.side_effect = mock_function_handing
    obj = NotifyMatrix(host='host', include_image=True, version='2')
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.access_token is None
    assert obj.notify('test', 'test') is False
    obj = NotifyMatrix(host='host', include_image=False, version='2')
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.access_token is None
    assert obj.notify('test', 'test') is True
    del obj

    def mock_function_handing(url, data, **kwargs):
        if False:
            return 10
        '\n        dummy function for handling image posts (successfully)\n        '
        response_obj = {'room_id': '!abc123:localhost', 'room_alias': '#abc123:localhost', 'joined_rooms': ['!abc123:localhost', '!def456:localhost'], 'access_token': 'abcd1234', 'home_server': 'localhost'}
        request = mock.Mock()
        request.content = dumps(response_obj)
        request.status_code = requests.codes.ok
        return request
    mock_get.side_effect = mock_function_handing
    mock_put.side_effect = mock_function_handing
    mock_post.side_effect = mock_function_handing
    obj = NotifyMatrix(host='host', include_image=True)
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.access_token is None
    assert obj.notify('test', 'test') is True
    obj = NotifyMatrix(host='host', include_image=False)
    assert isinstance(obj, NotifyMatrix) is True
    assert obj.access_token is None
    assert obj.notify('test', 'test') is True
    del obj

@mock.patch('requests.put')
@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_matrix_attachments_api_v3(mock_post, mock_get, mock_put):
    if False:
        i = 10
        return i + 15
    '\n    NotifyMatrix() Attachment Checks (v3)\n\n    '
    response = mock.Mock()
    response.status_code = requests.codes.ok
    response.content = MATRIX_GOOD_RESPONSE.encode('utf-8')
    bad_response = mock.Mock()
    bad_response.status_code = requests.codes.internal_server_error
    mock_post.return_value = response
    mock_get.return_value = response
    mock_put.return_value = response
    obj = Apprise.instantiate('matrix://user:pass@localhost/#general?v=3')
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert mock_put.call_count == 1
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0][0][0] == 'http://localhost/_matrix/client/v3/login'
    assert mock_post.call_args_list[1][0][0] == 'http://localhost/_matrix/client/v3/join/%23general%3Alocalhost'
    assert mock_put.call_args_list[0][0][0] == 'http://localhost/_matrix/client/v3/rooms/%21abc123%3Alocalhost/send/m.room.message/0'
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-archive.zip'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    mock_post.return_value = None
    for side_effect in (requests.RequestException(), OSError(), bad_response):
        mock_post.side_effect = [side_effect]
        assert obj.send(body='test', attach=attach) is True
    for side_effect in (requests.RequestException(), OSError(), bad_response):
        mock_post.side_effect = [response, side_effect]
        assert obj.send(body='test', attach=attach) is True
    mock_post.side_effect = [response, bad_response, response]
    assert obj.send(body='test', attach=attach) is True
    del obj

@mock.patch('requests.get')
@mock.patch('requests.post')
def test_plugin_matrix_attachments_api_v2(mock_post, mock_get):
    if False:
        print('Hello World!')
    '\n    NotifyMatrix() Attachment Checks (v2)\n\n    '
    response = mock.Mock()
    response.status_code = requests.codes.ok
    response.content = MATRIX_GOOD_RESPONSE.encode('utf-8')
    bad_response = mock.Mock()
    bad_response.status_code = requests.codes.internal_server_error
    mock_post.return_value = response
    mock_get.return_value = response
    obj = Apprise.instantiate('matrix://user:pass@localhost/#general?v=2')
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    mock_post.return_value = response
    mock_get.return_value = response
    mock_post.side_effect = None
    mock_get.side_effect = None
    del obj
    obj = Apprise.instantiate('matrixs://user:pass@localhost/#general?v=2')
    mock_post.reset_mock()
    mock_get.reset_mock()
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    assert mock_post.call_count == 5
    assert mock_post.call_args_list[0][0][0] == 'https://localhost/_matrix/client/r0/login'
    assert mock_post.call_args_list[1][0][0] == 'https://localhost/_matrix/media/r0/upload'
    assert mock_post.call_args_list[2][0][0] == 'https://localhost/_matrix/client/r0/join/%23general%3Alocalhost'
    assert mock_post.call_args_list[3][0][0] == 'https://localhost/_matrix/client/r0/rooms/%21abc123%3Alocalhost/send/m.room.message'
    assert mock_post.call_args_list[4][0][0] == 'https://localhost/_matrix/client/r0/rooms/%21abc123%3Alocalhost/send/m.room.message'
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-archive.zip'))
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=attach) is True
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    attach = AppriseAttachment(path)
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO, attach=path) is False
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    mock_post.return_value = None
    mock_get.return_value = None
    for side_effect in (requests.RequestException(), OSError(), bad_response):
        mock_post.reset_mock()
        mock_get.reset_mock()
        mock_post.side_effect = [side_effect, response]
        mock_get.side_effect = [side_effect, response]
        assert obj.send(body='test', attach=attach) is False
    for side_effect in (requests.RequestException(), OSError(), bad_response):
        mock_post.reset_mock()
        mock_get.reset_mock()
        mock_post.side_effect = [response, side_effect, side_effect, response]
        mock_get.side_effect = [side_effect, side_effect, response]
        assert obj.send(body='test', attach=attach) is False
    mock_post.side_effect = [response, bad_response, response, response, response, response]
    mock_get.side_effect = [response, bad_response, response, response, response, response]
    assert obj.send(body='test', attach=attach) is False
    del obj
    obj = Apprise.instantiate('matrixs://user:pass@localhost/#general?v=2&image=y')
    mock_post.reset_mock()
    mock_get.reset_mock()
    mock_post.return_value = None
    mock_get.return_value = None
    mock_post.side_effect = [response, response, bad_response, response, response, response, response]
    mock_get.side_effect = [response, response, bad_response, response, response, response, response]
    assert obj.notify(body='body', title='title', notify_type=NotifyType.INFO) is False
    mock_post.return_value = response
    mock_get.return_value = response
    mock_post.side_effect = None
    mock_get.side_effect = None
    assert obj.send(body='test', attach=attach) is True
    del obj
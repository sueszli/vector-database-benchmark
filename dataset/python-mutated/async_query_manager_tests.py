from unittest.mock import Mock
from jwt import encode
from pytest import fixture, raises
from superset.async_events.async_query_manager import AsyncQueryManager, AsyncQueryTokenException
JWT_TOKEN_SECRET = 'some_secret'
JWT_TOKEN_COOKIE_NAME = 'superset_async_jwt'

@fixture
def async_query_manager():
    if False:
        print('Hello World!')
    query_manager = AsyncQueryManager()
    query_manager._jwt_secret = JWT_TOKEN_SECRET
    query_manager._jwt_cookie_name = JWT_TOKEN_COOKIE_NAME
    return query_manager

def test_parse_channel_id_from_request(async_query_manager):
    if False:
        return 10
    encoded_token = encode({'channel': 'test_channel_id'}, JWT_TOKEN_SECRET, algorithm='HS256')
    request = Mock()
    request.cookies = {'superset_async_jwt': encoded_token}
    assert async_query_manager.parse_channel_id_from_request(request) == 'test_channel_id'

def test_parse_channel_id_from_request_no_cookie(async_query_manager):
    if False:
        while True:
            i = 10
    request = Mock()
    request.cookies = {}
    with raises(AsyncQueryTokenException):
        async_query_manager.parse_channel_id_from_request(request)

def test_parse_channel_id_from_request_bad_jwt(async_query_manager):
    if False:
        while True:
            i = 10
    request = Mock()
    request.cookies = {'superset_async_jwt': 'bad_jwt'}
    with raises(AsyncQueryTokenException):
        async_query_manager.parse_channel_id_from_request(request)
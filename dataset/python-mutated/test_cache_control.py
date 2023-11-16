from core.middleware.cache_control import NeverCacheMiddleware
from django.http import HttpResponse

def test_NoCacheMiddleware_adds_cache_control_headers(mocker):
    if False:
        i = 10
        return i + 15
    a_response = HttpResponse()
    mocked_get_response = mocker.MagicMock(return_value=a_response)
    mock_request = mocker.MagicMock()
    middleware = NeverCacheMiddleware(mocked_get_response)
    response = middleware(mock_request)
    assert response.headers['Cache-Control'] == 'max-age=0, no-cache, no-store, must-revalidate, private'
    assert response.headers['Pragma'] == 'no-cache'
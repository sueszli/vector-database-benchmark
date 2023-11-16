from google.appengine.api import urlfetch
import mock
import pytest
import webtest
import rpc

@pytest.fixture
def app():
    if False:
        print('Hello World!')
    return webtest.TestApp(rpc.app)

@mock.patch('rpc.urlfetch')
def test_url_fetch(urlfetch_mock, app):
    if False:
        i = 10
        return i + 15
    get_result_mock = mock.Mock(return_value=mock.Mock(status_code=200, content="I'm Feeling Lucky"))
    urlfetch_mock.create_rpc = mock.Mock(return_value=mock.Mock(get_result=get_result_mock))
    response = app.get('/')
    assert response.status_int == 200
    assert "I'm Feeling Lucky" in response.body

@mock.patch('rpc.urlfetch')
def test_url_fetch_rpc_error(urlfetch_mock, app):
    if False:
        while True:
            i = 10
    urlfetch_mock.DownloadError = urlfetch.DownloadError
    get_result_mock = mock.Mock(side_effect=urlfetch.DownloadError())
    urlfetch_mock.create_rpc = mock.Mock(return_value=mock.Mock(get_result=get_result_mock))
    response = app.get('/', status=500)
    assert 'Error fetching URL' in response.body

@mock.patch('rpc.urlfetch')
def test_url_fetch_http_error(urlfetch_mock, app):
    if False:
        for i in range(10):
            print('nop')
    get_result_mock = mock.Mock(return_value=mock.Mock(status_code=404, content='Not Found'))
    urlfetch_mock.create_rpc = mock.Mock(return_value=mock.Mock(get_result=get_result_mock))
    response = app.get('/', status=404)
    assert '404' in response.body

@mock.patch('rpc.urlfetch')
def test_url_post(urlfetch_mock, app):
    if False:
        i = 10
        return i + 15
    get_result_mock = mock.Mock(return_value=mock.Mock(status_code=200, content="I'm Feeling Lucky"))
    urlfetch_mock.create_rpc = mock.Mock(return_value=mock.Mock(get_result=get_result_mock))
    rpc_mock = mock.Mock()
    urlfetch_mock.create_rpc.return_value = rpc_mock
    app.get('/callback')
    rpc_mock.wait.assert_called_with()
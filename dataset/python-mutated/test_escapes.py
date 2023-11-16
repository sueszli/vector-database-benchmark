from __future__ import print_function
from json import loads
import requests
from unittest import mock
import apprise

@mock.patch('requests.post')
def test_apprise_interpret_escapes(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    API: Apprise() interpret-escape tests\n    '
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    asset = apprise.AppriseAsset()
    assert asset.interpret_escapes is False
    a = apprise.Apprise(asset=asset)
    assert a.add('json://localhost') is True
    a[0].asset.interpret_escapes is False
    assert a.notify('ab\\ncd') is True
    assert mock_post.call_count == 1
    loads(mock_post.call_args_list[0][1]['data']).get('message', '') == 'ab\\ncd'
    mock_post.reset_mock()
    assert a.notify('ab\\ncd', interpret_escapes=True) is True
    assert mock_post.call_count == 1
    loads(mock_post.call_args_list[0][1]['data']).get('message', '') == 'ab\ncd'
    mock_post.reset_mock()
    asset = apprise.AppriseAsset(interpret_escapes=True)
    assert asset.interpret_escapes is True
    a = apprise.Apprise(asset=asset)
    assert a.add('json://localhost') is True
    a[0].asset.interpret_escapes is True
    assert a.notify('ab\\ncd') is True
    assert mock_post.call_count == 1
    loads(mock_post.call_args_list[0][1]['data']).get('message', '') == 'ab\ncd'
    mock_post.reset_mock()
    assert a.notify('ab\\ncd', interpret_escapes=False) is True
    assert mock_post.call_count == 1
    loads(mock_post.call_args_list[0][1]['data']).get('message', '') == 'ab\\ncd'

@mock.patch('requests.post')
def test_apprise_escaping(mock_post):
    if False:
        return 10
    '\n    API: Apprise() escaping tests\n\n    '
    a = apprise.Apprise()
    response = mock.Mock()
    response.content = ''
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    a.add('json://localhost')
    assert a.notify(title='\\r\\ntitle\\r\\n', body='\\r\\nbody\\r\\n', interpret_escapes=True)
    assert mock_post.call_count == 1
    result = loads(mock_post.call_args_list[0][1]['data'])
    assert result['title'] == 'title'
    assert result['message'] == '\r\nbody'
    mock_post.reset_mock()
    assert a.notify(title='دعونا نجعل العالم مكانا أفضل.\\r\\t\\t\\n\\r\\n', body='Egy sor kódot egyszerre.\\r\\n\\r\\r\\n', interpret_escapes=True)
    assert mock_post.call_count == 1
    result = loads(mock_post.call_args_list[0][1]['data'])
    assert result['title'] == 'دعونا نجعل العالم مكانا أفضل.'
    assert result['message'] == 'Egy sor kódot egyszerre.'
    assert a.notify(title=None, body=4, interpret_escapes=True) is False
    assert a.notify(title=4, body=None, interpret_escapes=True) is False
    assert a.notify(title=object(), body=False, interpret_escapes=True) is False
    assert a.notify(title=False, body=object(), interpret_escapes=True) is False
    assert a.notify(title=b'byte title', body=b'byte body', interpret_escapes=True) is True
    title = 'כותרת נפלאה'.encode('ISO-8859-8')
    body = '[_[זו הודעה](http://localhost)_'.encode('ISO-8859-8')
    assert a.notify(title=title, body=body, interpret_escapes=True) is False
    asset = apprise.AppriseAsset(encoding='ISO-8859-8')
    a = apprise.Apprise(asset=asset)
    a.add('json://localhost')
    assert a.notify(title=title, body=body, interpret_escapes=True) is True
    a = apprise.Apprise()
    a.add('json://localhost')
    assert a.notify(title=None, body='valid', interpret_escapes=True) is True
    assert a.notify(title=4, body='valid', interpret_escapes=True) is False
    assert a.notify(title=object(), body='valid', interpret_escapes=True) is False
    assert a.notify(title=False, body='valid', interpret_escapes=True) is True
    assert a.notify(title=b'byte title', body='valid', interpret_escapes=True) is True
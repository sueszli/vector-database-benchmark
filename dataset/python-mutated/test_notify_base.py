import pytest
from datetime import datetime
from datetime import timedelta
from apprise.plugins.NotifyBase import NotifyBase
from apprise import NotifyType
from apprise import NotifyImageSize
from timeit import default_timer
import logging
logging.disable(logging.CRITICAL)

def test_notify_base():
    if False:
        i = 10
        return i + 15
    '\n    API: NotifyBase() object\n\n    '
    with pytest.raises(TypeError):
        NotifyBase(**{'format': 'invalid'})
    with pytest.raises(TypeError):
        NotifyBase(**{'overflow': 'invalid'})
    nb = NotifyBase(port='invalid')
    assert nb.port is None
    nb = NotifyBase(port=10)
    assert nb.port == 10
    assert isinstance(nb.url(), str)
    assert str(nb) == nb.url()
    try:
        nb.send('test message')
        assert False
    except NotImplementedError:
        assert True
    nb = NotifyBase()
    nb.request_rate_per_sec = 0.0
    start_time = default_timer()
    nb.throttle()
    elapsed = default_timer() - start_time
    assert elapsed < 0.5
    start_time = default_timer()
    nb.throttle()
    elapsed = default_timer() - start_time
    assert elapsed < 0.5
    nb = NotifyBase()
    nb.request_rate_per_sec = 1.0
    start_time = default_timer()
    nb.throttle()
    elapsed = default_timer() - start_time
    assert elapsed < 0.5
    start_time = default_timer()
    nb.throttle(last_io=datetime.now())
    elapsed = default_timer() - start_time
    assert elapsed > 0.48 and elapsed < 1.5
    nb = NotifyBase()
    nb.request_rate_per_sec = 1.0
    start_time = default_timer()
    nb.throttle(last_io=datetime.now())
    elapsed = default_timer() - start_time
    assert elapsed > 0.48 and elapsed < 1.5
    start_time = default_timer()
    nb.throttle(last_io=datetime.now())
    elapsed = default_timer() - start_time
    assert elapsed > 0.48 and elapsed < 1.5
    nb = NotifyBase()
    start_time = default_timer()
    nb.request_rate_per_sec = 1.0
    nb.throttle(last_io=datetime.now() - timedelta(seconds=20))
    elapsed = default_timer() - start_time
    assert elapsed < 0.5
    start_time = default_timer()
    nb.throttle(wait=0.5)
    elapsed = default_timer() - start_time
    assert elapsed > 0.48 and elapsed < 1.5
    assert nb.image_url(notify_type=NotifyType.INFO) is None
    assert nb.image_path(notify_type=NotifyType.INFO) is None
    assert nb.image_raw(notify_type=NotifyType.INFO) is None
    assert nb.color(notify_type='invalid') is None
    assert isinstance(nb.color(notify_type=NotifyType.INFO, color_type=None), str)
    assert isinstance(nb.color(notify_type=NotifyType.INFO, color_type=int), int)
    assert isinstance(nb.color(notify_type=NotifyType.INFO, color_type=tuple), tuple)
    nb = NotifyBase()
    nb.image_size = NotifyImageSize.XY_256
    assert nb.image_url(notify_type=NotifyType.INFO) is not None
    assert nb.image_path(notify_type=NotifyType.INFO) is not None
    assert nb.image_raw(notify_type=NotifyType.INFO) is not None
    assert nb.image_url(notify_type='invalid') is None
    assert nb.image_path(notify_type='invalid') is None
    assert nb.image_raw(notify_type='invalid') is None
    assert NotifyBase.escape_html("<content>'\t \n</content>") == '&lt;content&gt;&apos;&emsp;&nbsp;\n&lt;/content&gt;'
    assert NotifyBase.escape_html("<content>'\t \n</content>", convert_new_lines=True) == '&lt;content&gt;&apos;&emsp;&nbsp;<br/>&lt;/content&gt;'
    assert NotifyBase.split_path(None) == []
    assert NotifyBase.split_path(object()) == []
    assert NotifyBase.split_path(42) == []
    assert NotifyBase.split_path('/path/?name=Dr%20Disrespect', unquote=False) == ['path', '?name=Dr%20Disrespect']
    assert NotifyBase.split_path('/path/?name=Dr%20Disrespect', unquote=True) == ['path', '?name=Dr Disrespect']
    assert NotifyBase.split_path('/%2F///%2F%2F////%2F%2F%2F////', unquote=True) == ['/', '//', '///']
    assert NotifyBase.parse_list(None) == []
    assert NotifyBase.parse_list(object()) == []
    assert NotifyBase.parse_list(42) == []
    result = NotifyBase.parse_list(',path,?name=Dr%20Disrespect', unquote=False)
    assert isinstance(result, list) is True
    assert len(result) == 2
    assert 'path' in result
    assert '?name=Dr%20Disrespect' in result
    result = NotifyBase.parse_list(',path,?name=Dr%20Disrespect', unquote=True)
    assert isinstance(result, list) is True
    assert len(result) == 2
    assert 'path' in result
    assert '?name=Dr Disrespect' in result
    result = NotifyBase.parse_list(',%2F,%2F%2F, , , ,%2F%2F%2F, %2F', unquote=True)
    assert isinstance(result, list) is True
    assert len(result) == 3
    assert '/' in result
    assert '//' in result
    assert '///' in result
    assert NotifyBase.parse_phone_no(None) == []
    assert NotifyBase.parse_phone_no(object()) == []
    assert NotifyBase.parse_phone_no(42) == []
    result = NotifyBase.parse_phone_no('+1-800-123-1234,(800) 123-4567', unquote=False)
    assert isinstance(result, list) is True
    assert len(result) == 2
    assert '+1-800-123-1234' in result
    assert '(800) 123-4567' in result
    result = NotifyBase.parse_phone_no('%2B1-800-123-1234,%2B1%20800%20123%204567', unquote=True)
    assert isinstance(result, list) is True
    assert len(result) == 2
    assert '+1-800-123-1234' in result
    assert '+1 800 123 4567' in result
    assert NotifyBase.escape_html('') == ''
    assert NotifyBase.escape_html(None) == ''
    assert NotifyBase.escape_html(object()) == ''
    assert NotifyBase.unquote('%20') == ' '
    assert NotifyBase.quote(' ') == '%20'
    assert NotifyBase.unquote(None) == ''
    assert NotifyBase.quote(None) == ''

def test_notify_base_urls():
    if False:
        i = 10
        return i + 15
    '\n    API: NotifyBase() URLs\n\n    '
    results = NotifyBase.parse_url('https://localhost:8080/?verify=No')
    assert 'verify' in results
    assert results['verify'] is False
    results = NotifyBase.parse_url('https://localhost:8080/?verify=Yes')
    assert 'verify' in results
    assert results['verify'] is True
    results = NotifyBase.parse_url('https://localhost:8080')
    assert 'verify' in results
    assert results['verify'] is True
    results = NotifyBase.parse_url('https://user:pass@localhost')
    assert 'password' in results
    assert results['password'] == 'pass'
    results = NotifyBase.parse_url('https://user:pass@localhost?pass=newpassword')
    assert 'password' in results
    assert results['password'] == 'newpassword'
    results = NotifyBase.parse_url('https://user:pass@localhost?password=passwd')
    assert 'password' in results
    assert results['password'] == 'passwd'
    results = NotifyBase.parse_url('https://user:pass@localhost?pass=pw1&password=pw2')
    assert 'password' in results
    assert results['password'] == 'pw1'
    results = NotifyBase.parse_url('https://localhost?format=invalid')
    assert 'format' not in results
    results = NotifyBase.parse_url('https://localhost?format=text')
    assert 'format' in results
    assert results['format'] == 'text'
    results = NotifyBase.parse_url('https://localhost?format=markdown')
    assert 'format' in results
    assert results['format'] == 'markdown'
    results = NotifyBase.parse_url('https://localhost?format=html')
    assert 'format' in results
    assert results['format'] == 'html'
    results = NotifyBase.parse_url('https://localhost?overflow=invalid')
    assert 'overflow' not in results
    results = NotifyBase.parse_url('https://localhost?overflow=upstream')
    assert 'overflow' in results
    assert results['overflow'] == 'upstream'
    results = NotifyBase.parse_url('https://localhost?overflow=split')
    assert 'overflow' in results
    assert results['overflow'] == 'split'
    results = NotifyBase.parse_url('https://localhost?overflow=truncate')
    assert 'overflow' in results
    assert results['overflow'] == 'truncate'
    results = NotifyBase.parse_url('https://user:pass@localhost')
    assert 'user' in results
    assert results['user'] == 'user'
    results = NotifyBase.parse_url('https://user:pass@localhost?user=newuser')
    assert 'user' in results
    assert results['user'] == 'newuser'
    assert NotifyBase.parse_url('https://:@/') is None
    assert NotifyBase.parse_url('http://:@') is None
    assert NotifyBase.parse_url('http://@') is None
    assert NotifyBase.parse_url('http:///') is None
    assert NotifyBase.parse_url('http://:test/') is None
    assert NotifyBase.parse_url('http://pass:test/') is None
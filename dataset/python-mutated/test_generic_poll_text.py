import sys
from importlib import reload
from types import ModuleType
import pytest
from libqtile.widget import generic_poll_text

class Mockxml(ModuleType):

    @classmethod
    def parse(cls, value):
        if False:
            i = 10
            return i + 15
        return {'test': value}

class MockRequest:
    return_value = None

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        pass

class Mockurlopen:

    def __init__(self, request):
        if False:
            print('Hello World!')
        self.request = request

    class headers:

        @classmethod
        def get_content_charset(cls):
            if False:
                while True:
                    i = 10
            return 'utf-8'

    def read(self):
        if False:
            i = 10
            return i + 15
        return self.request.return_value

def test_gen_poll_text():
    if False:
        print('Hello World!')
    gpt_no_func = generic_poll_text.GenPollText()
    assert gpt_no_func.poll() == 'You need a poll function'
    gpt_with_func = generic_poll_text.GenPollText(func=lambda : 'Has function')
    assert gpt_with_func.poll() == 'Has function'

def test_gen_poll_url_not_configured():
    if False:
        for i in range(10):
            print('nop')
    gpurl = generic_poll_text.GenPollUrl()
    assert gpurl.poll() == 'Invalid config'

def test_gen_poll_url_no_json():
    if False:
        print('Hello World!')
    gpurl = generic_poll_text.GenPollUrl(json=False)
    assert 'Content-Type' not in gpurl.headers

def test_gen_poll_url_headers_and_json():
    if False:
        i = 10
        return i + 15
    gpurl = generic_poll_text.GenPollUrl(headers={'fake-header': 'fake-value'}, data={'argument': 'data value'}, user_agent='qtile test')
    assert gpurl.headers['User-agent'] == 'qtile test'
    assert gpurl.headers['fake-header'] == 'fake-value'
    assert gpurl.headers['Content-Type'] == 'application/json'
    assert gpurl.data.decode() == '{"argument": "data value"}'

def test_gen_poll_url_text(monkeypatch):
    if False:
        return 10
    gpurl = generic_poll_text.GenPollUrl(json=False, parse=lambda x: x, url='testing')
    monkeypatch.setattr(generic_poll_text, 'Request', MockRequest)
    monkeypatch.setattr(generic_poll_text, 'urlopen', Mockurlopen)
    generic_poll_text.Request.return_value = b'OK'
    assert gpurl.poll() == 'OK'

def test_gen_poll_url_json(monkeypatch):
    if False:
        i = 10
        return i + 15
    gpurl = generic_poll_text.GenPollUrl(parse=lambda x: x, data=[1, 2, 3], url='testing')
    monkeypatch.setattr(generic_poll_text, 'Request', MockRequest)
    monkeypatch.setattr(generic_poll_text, 'urlopen', Mockurlopen)
    generic_poll_text.Request.return_value = b'{"test": "OK"}'
    assert gpurl.poll()['test'] == 'OK'

def test_gen_poll_url_xml_no_xmltodict(monkeypatch):
    if False:
        i = 10
        return i + 15
    gpurl = generic_poll_text.GenPollUrl(json=False, xml=True, parse=lambda x: x, url='testing')
    monkeypatch.setattr(generic_poll_text, 'Request', MockRequest)
    monkeypatch.setattr(generic_poll_text, 'urlopen', Mockurlopen)
    generic_poll_text.Request.return_value = b'OK'
    with pytest.raises(Exception):
        gpurl.poll()

def test_gen_poll_url_xml_has_xmltodict(monkeypatch):
    if False:
        return 10
    monkeypatch.setitem(sys.modules, 'xmltodict', Mockxml('xmltodict'))
    reload(generic_poll_text)
    gpurl = generic_poll_text.GenPollUrl(json=False, xml=True, parse=lambda x: x, url='testing')
    monkeypatch.setattr(generic_poll_text, 'Request', MockRequest)
    monkeypatch.setattr(generic_poll_text, 'urlopen', Mockurlopen)
    generic_poll_text.Request.return_value = b'OK'
    assert gpurl.poll()['test'] == 'OK'

def test_gen_poll_url_broken_parse(monkeypatch):
    if False:
        print('Hello World!')
    gpurl = generic_poll_text.GenPollUrl(json=False, parse=lambda x: x.foo, url='testing')
    monkeypatch.setattr(generic_poll_text, 'Request', MockRequest)
    monkeypatch.setattr(generic_poll_text, 'urlopen', Mockurlopen)
    generic_poll_text.Request.return_value = b'OK'
    assert gpurl.poll() == "Can't parse"
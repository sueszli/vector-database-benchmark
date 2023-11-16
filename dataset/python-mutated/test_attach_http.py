import re
from unittest import mock
import requests
import mimetypes
from os.path import join
from os.path import dirname
from os.path import getsize
from apprise.attachment.AttachHTTP import AttachHTTP
from apprise import AppriseAttachment
from apprise.plugins.NotifyBase import NotifyBase
from apprise.common import NOTIFY_SCHEMA_MAP
from apprise.common import ContentLocation
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = join(dirname(__file__), 'var')
REQUEST_EXCEPTIONS = (requests.ConnectionError(0, 'requests.ConnectionError() not handled'), requests.RequestException(0, 'requests.RequestException() not handled'), requests.HTTPError(0, 'requests.HTTPError() not handled'), requests.ReadTimeout(0, 'requests.ReadTimeout() not handled'), requests.TooManyRedirects(0, 'requests.TooManyRedirects() not handled'), OSError('SystemError'))

def test_attach_http_parse_url():
    if False:
        for i in range(10):
            print('nop')
    '\n    API: AttachHTTP().parse_url()\n\n    '
    assert AttachHTTP.parse_url('garbage://') is None
    assert AttachHTTP.parse_url('http://') is None

def test_attach_http_query_string_dictionary():
    if False:
        return 10
    '\n    API: AttachHTTP() Query String Dictionary\n\n    '
    results = AttachHTTP.parse_url('http://localhost')
    assert isinstance(results, dict)
    obj = AttachHTTP(**results)
    assert isinstance(obj, AttachHTTP)
    assert re.search('[?&]verify=yes', obj.url())
    results = AttachHTTP.parse_url('http://localhost?dl=1&_var=test')
    assert isinstance(results, dict)
    obj = AttachHTTP(**results)
    assert isinstance(obj, AttachHTTP)
    assert re.search('[?&]verify=yes', obj.url())
    assert re.search('[?&]dl=1', obj.url())
    assert re.search('[?&]_var=test', obj.url())

@mock.patch('requests.get')
def test_attach_http(mock_get):
    if False:
        return 10
    '\n    API: AttachHTTP() object\n\n    '

    class GoodNotification(NotifyBase):

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            super().__init__(*args, **kwargs)

        def notify(self, *args, **kwargs):
            if False:
                return 10
            return True

        def url(self):
            if False:
                while True:
                    i = 10
            return ''
    NOTIFY_SCHEMA_MAP['good'] = GoodNotification
    path = join(TEST_VAR_DIR, 'apprise-test.gif')

    class DummyResponse:
        """
        A dummy response used to manage our object
        """
        status_code = requests.codes.ok
        headers = {'Content-Length': getsize(path), 'Content-Type': 'image/gif'}
        ptr = None
        _keepalive_chunk_ref = 0

        def close(self):
            if False:
                for i in range(10):
                    print('nop')
            return

        def iter_content(self, chunk_size=1024):
            if False:
                return 10
            'Lazy function (generator) to read a file piece by piece.\n            Default chunk size: 1k.'
            while True:
                self._keepalive_chunk_ref += 1
                if 16 % self._keepalive_chunk_ref == 0:
                    yield ''
                data = self.ptr.read(chunk_size)
                if not data:
                    break
                yield data

        def raise_for_status(self):
            if False:
                i = 10
                return i + 15
            return

        def __enter__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.ptr = open(path, 'rb')
            return self

        def __exit__(self, *args, **kwargs):
            if False:
                return 10
            self.ptr.close()
    dummy_response = DummyResponse()
    mock_get.return_value = dummy_response
    results = AttachHTTP.parse_url('http://user:pass@localhost/apprise.gif?dl=1&cache=300')
    assert isinstance(results, dict)
    attachment = AttachHTTP(**results)
    assert isinstance(attachment.url(), str) is True
    assert mock_get.call_count == 0
    assert attachment
    assert mock_get.call_count == 1
    assert 'params' in mock_get.call_args_list[0][1]
    assert 'dl' in mock_get.call_args_list[0][1]['params']
    assert 'cache' not in mock_get.call_args_list[0][1]['params']
    results = AttachHTTP.parse_url('http://user:pass@localhost/apprise.gif?+key=value&cache=True')
    assert isinstance(results, dict)
    attachment = AttachHTTP(**results)
    assert isinstance(attachment.url(), str) is True
    assert re.search('[?&]mime=', attachment.url()) is None
    assert re.search('[?&]name=', attachment.url()) is None
    assert attachment.name == 'apprise.gif'
    assert attachment.mimetype == 'image/gif'
    results = AttachHTTP.parse_url('http://localhost:3000/noname.gif?name=usethis.jpg&mime=image/jpeg')
    assert isinstance(results, dict)
    attachment = AttachHTTP(**results)
    assert isinstance(attachment.url(), str) is True
    assert re.search('[?&]mime=image%2Fjpeg', attachment.url())
    assert re.search('[?&]name=usethis.jpg', attachment.url())
    assert attachment.name == 'usethis.jpg'
    assert attachment.mimetype == 'image/jpeg'
    assert attachment.download()
    assert attachment
    assert len(attachment) == getsize(path)
    attachment = AttachHTTP(**results)
    attachment.location = ContentLocation.INACCESSIBLE
    assert attachment.path is None
    assert attachment.download() is False
    results = AttachHTTP.parse_url('http://localhost')
    assert isinstance(results, dict)
    attachment = AttachHTTP(**results)
    assert isinstance(attachment.url(), str) is True
    assert re.search('[?&]mime=', attachment.url()) is None
    assert re.search('[?&]name=', attachment.url()) is None
    assert attachment.mimetype == 'image/gif'
    assert attachment.name == '{}{}'.format(AttachHTTP.unknown_filename, mimetypes.guess_extension(attachment.mimetype))
    assert attachment
    assert len(attachment) == getsize(path)
    dummy_response.headers['Content-Length'] = AttachHTTP.max_file_size + 1
    results = AttachHTTP.parse_url('http://localhost/toobig.jpg')
    assert isinstance(results, dict)
    attachment = AttachHTTP(**results)
    assert not attachment
    assert isinstance(attachment.url(), str) is True
    assert re.search('[?&]mime=', attachment.url()) is None
    assert re.search('[?&]name=', attachment.url()) is None
    assert attachment.mimetype is None
    assert attachment.name is None
    assert len(attachment) == 0
    del dummy_response.headers['Content-Length']
    results = AttachHTTP.parse_url('http://localhost/no-length.gif')
    assert isinstance(results, dict)
    attachment = AttachHTTP(**results)
    assert isinstance(attachment.url(), str) is True
    assert re.search('[?&]mime=', attachment.url()) is None
    assert re.search('[?&]name=', attachment.url()) is None
    assert attachment.mimetype == 'image/gif'
    assert attachment.name == 'no-length.gif'
    assert attachment
    assert len(attachment) == getsize(path)
    max_file_size = AttachHTTP.max_file_size
    AttachHTTP.max_file_size = getsize(path)
    dummy_response.headers['Content-Disposition'] = 'attachment; filename="myimage.gif"'
    del dummy_response.headers['Content-Type']
    results = AttachHTTP.parse_url('http://user@localhost/ignore-filename.gif')
    assert isinstance(results, dict)
    attachment = AttachHTTP(**results)
    assert isinstance(attachment.url(), str) is True
    assert re.search('[?&]mime=', attachment.url()) is None
    assert re.search('[?&]name=', attachment.url()) is None
    assert attachment.mimetype == 'image/gif'
    assert attachment.name == 'myimage.gif'
    assert attachment
    assert len(attachment) == getsize(path)
    AttachHTTP.max_file_size = getsize(path) - 1
    results = AttachHTTP.parse_url('http://localhost/toobig.jpg')
    assert isinstance(results, dict)
    attachment = AttachHTTP(**results)
    assert not attachment
    assert isinstance(attachment.url(), str) is True
    assert re.search('[?&]mime=', attachment.url()) is None
    assert re.search('[?&]name=', attachment.url()) is None
    assert attachment.mimetype is None
    assert attachment.name is None
    assert len(attachment) == 0
    AttachHTTP.max_file_size = 0
    results = AttachHTTP.parse_url('http://user@localhost')
    assert isinstance(results, dict)
    attachment = AttachHTTP(**results)
    assert isinstance(attachment.url(), str) is True
    assert re.search('[?&]mime=', attachment.url()) is None
    assert re.search('[?&]name=', attachment.url()) is None
    assert attachment.mimetype == 'image/gif'
    assert attachment.name == 'myimage.gif'
    assert attachment
    assert len(attachment) == getsize(path)
    dummy_response.headers = {'Content-Length': 'invalid'}
    results = AttachHTTP.parse_url('http://localhost/invalid-length.gif')
    assert isinstance(results, dict)
    attachment = AttachHTTP(**results)
    assert isinstance(attachment.url(), str) is True
    assert re.search('[?&]mime=', attachment.url()) is None
    assert re.search('[?&]name=', attachment.url()) is None
    assert attachment.mimetype == 'image/gif'
    assert attachment.name == 'invalid-length.gif'
    assert attachment
    dummy_response.headers = {}
    results = AttachHTTP.parse_url('http://user@localhost')
    assert isinstance(results, dict)
    attachment = AttachHTTP(**results)
    assert attachment
    assert isinstance(attachment.url(), str) is True
    assert re.search('[?&]mime=', attachment.url()) is None
    assert re.search('[?&]name=', attachment.url()) is None
    attachment.detected_name = None
    assert attachment.mimetype == attachment.unknown_mimetype
    assert attachment.name.startswith(AttachHTTP.unknown_filename)
    assert len(attachment) == getsize(path)
    mock_get.return_value = None
    for _exception in REQUEST_EXCEPTIONS:
        aa = AppriseAttachment.instantiate('http://localhost/exception.gif?cache=30')
        assert isinstance(aa, AttachHTTP)
        mock_get.side_effect = _exception
        assert not aa
    AttachHTTP.max_file_size = max_file_size
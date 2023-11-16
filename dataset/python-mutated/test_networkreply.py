"""Tests for browser.network.networkreply."""
import pytest
from qutebrowser.qt.core import QUrl, QIODevice
from qutebrowser.qt.network import QNetworkRequest, QNetworkReply
from qutebrowser.browser.webkit.network import networkreply

@pytest.fixture
def req():
    if False:
        return 10
    return QNetworkRequest(QUrl('http://www.qutebrowser.org/'))

class TestFixedDataNetworkReply:

    def test_attributes(self, req):
        if False:
            print('Hello World!')
        reply = networkreply.FixedDataNetworkReply(req, b'', 'test/foo')
        assert reply.request() == req
        assert reply.url() == req.url()
        assert reply.openMode() == QIODevice.OpenModeFlag.ReadOnly
        assert reply.header(QNetworkRequest.KnownHeaders.ContentTypeHeader) == 'test/foo'
        http_code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        http_reason = reply.attribute(QNetworkRequest.Attribute.HttpReasonPhraseAttribute)
        assert http_code == 200
        assert http_reason == 'OK'
        assert reply.isFinished()
        assert not reply.isRunning()

    @pytest.mark.parametrize('data', [b'', b'foobar', b'Hello World! This is a test.'])
    def test_data(self, qtbot, req, data):
        if False:
            for i in range(10):
                print('nop')
        reply = networkreply.FixedDataNetworkReply(req, data, 'test/foo')
        with qtbot.wait_signals([reply.metaDataChanged, reply.readyRead, reply.finished], order='strict'):
            pass
        assert reply.bytesAvailable() == len(data)
        assert reply.readAll() == data

    @pytest.mark.parametrize('chunk_size', [1, 2, 3])
    def test_data_chunked(self, chunk_size, req):
        if False:
            for i in range(10):
                print('nop')
        data = b'123'
        reply = networkreply.FixedDataNetworkReply(req, data, 'test/foo')
        while data:
            assert reply.bytesAvailable() == len(data)
            assert reply.readData(chunk_size) == data[:chunk_size]
            data = data[chunk_size:]

    def test_abort(self, req):
        if False:
            for i in range(10):
                print('nop')
        reply = networkreply.FixedDataNetworkReply(req, b'foo', 'test/foo')
        reply.abort()
        assert reply.readAll() == b'foo'

def test_error_network_reply(qtbot, req):
    if False:
        print('Hello World!')
    reply = networkreply.ErrorNetworkReply(req, 'This is an error', QNetworkReply.NetworkError.UnknownNetworkError)
    with qtbot.wait_signals([reply.errorOccurred, reply.finished], order='strict'):
        pass
    reply.abort()
    assert reply.request() == req
    assert reply.url() == req.url()
    assert reply.openMode() == QIODevice.OpenModeFlag.ReadOnly
    assert reply.isFinished()
    assert not reply.isRunning()
    assert reply.bytesAvailable() == 0
    assert reply.readData(1) == b''
    assert reply.error() == QNetworkReply.NetworkError.UnknownNetworkError
    assert reply.errorString() == 'This is an error'

def test_redirect_network_reply():
    if False:
        return 10
    url = QUrl('https://www.example.com/')
    reply = networkreply.RedirectNetworkReply(url)
    assert reply.readData(1) == b''
    assert reply.attribute(QNetworkRequest.Attribute.RedirectionTargetAttribute) == url
    reply.abort()
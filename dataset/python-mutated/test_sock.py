from unittest import mock
from gunicorn import sock

@mock.patch('os.stat')
def test_create_sockets_unix_bytes(stat):
    if False:
        print('Hello World!')
    conf = mock.Mock(address=[b'127.0.0.1:8000'])
    log = mock.Mock()
    with mock.patch.object(sock.UnixSocket, '__init__', lambda *args: None):
        listeners = sock.create_sockets(conf, log)
        assert len(listeners) == 1
        print(type(listeners[0]))
        assert isinstance(listeners[0], sock.UnixSocket)

@mock.patch('os.stat')
def test_create_sockets_unix_strings(stat):
    if False:
        print('Hello World!')
    conf = mock.Mock(address=['127.0.0.1:8000'])
    log = mock.Mock()
    with mock.patch.object(sock.UnixSocket, '__init__', lambda *args: None):
        listeners = sock.create_sockets(conf, log)
        assert len(listeners) == 1
        assert isinstance(listeners[0], sock.UnixSocket)

def test_socket_close():
    if False:
        for i in range(10):
            print('nop')
    listener1 = mock.Mock()
    listener1.getsockname.return_value = ('127.0.0.1', '80')
    listener2 = mock.Mock()
    listener2.getsockname.return_value = ('192.168.2.5', '80')
    sock.close_sockets([listener1, listener2])
    listener1.close.assert_called_with()
    listener2.close.assert_called_with()

@mock.patch('os.unlink')
def test_unix_socket_close_unlink(unlink):
    if False:
        for i in range(10):
            print('nop')
    listener = mock.Mock()
    listener.getsockname.return_value = '/var/run/test.sock'
    sock.close_sockets([listener])
    listener.close.assert_called_with()
    unlink.assert_called_once_with('/var/run/test.sock')

@mock.patch('os.unlink')
def test_unix_socket_close_without_unlink(unlink):
    if False:
        for i in range(10):
            print('nop')
    listener = mock.Mock()
    listener.getsockname.return_value = '/var/run/test.sock'
    sock.close_sockets([listener], False)
    listener.close.assert_called_with()
    assert not unlink.called, 'unlink should not have been called'
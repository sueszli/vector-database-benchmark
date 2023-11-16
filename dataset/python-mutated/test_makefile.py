"""Tests for :py:mod:`cheroot.makefile`."""
from cheroot import makefile

class MockSocket:
    """A mock socket."""

    def __init__(self):
        if False:
            while True:
                i = 10
        'Initialize :py:class:`MockSocket`.'
        self.messages = []

    def recv_into(self, buf):
        if False:
            return 10
        'Simulate ``recv_into`` for Python 3.'
        if not self.messages:
            return 0
        msg = self.messages.pop(0)
        for (index, byte) in enumerate(msg):
            buf[index] = byte
        return len(msg)

    def recv(self, size):
        if False:
            return 10
        'Simulate ``recv`` for Python 2.'
        try:
            return self.messages.pop(0)
        except IndexError:
            return ''

    def send(self, val):
        if False:
            print('Hello World!')
        'Simulate a send.'
        return len(val)

def test_bytes_read():
    if False:
        i = 10
        return i + 15
    'Reader should capture bytes read.'
    sock = MockSocket()
    sock.messages.append(b'foo')
    rfile = makefile.MakeFile(sock, 'r')
    rfile.read()
    assert rfile.bytes_read == 3

def test_bytes_written():
    if False:
        for i in range(10):
            print('nop')
    'Writer should capture bytes written.'
    sock = MockSocket()
    sock.messages.append(b'foo')
    wfile = makefile.MakeFile(sock, 'w')
    wfile.write(b'bar')
    assert wfile.bytes_written == 3
try:
    import select
    import ssl
    import io
    import binascii
except ImportError:
    print('SKIP')
    raise SystemExit
from micropython import const
_MP_STREAM_POLL_RD = const(1)
_MP_STREAM_POLL_WR = const(4)
_MP_STREAM_POLL_NVAL = const(32)
_MP_STREAM_POLL = const(3)
_MP_STREAM_CLOSE = const(4)
key = binascii.unhexlify(b'3082013b020100024100cc20643fd3d9c21a0acba4f48f61aadd675f52175a9dcf07fbef610a6a6ba14abb891745cd18a1d4c056580d8ff1a639460f867013c8391cdc9f2e573b0f872d0203010001024100bb17a54aeb3dd7ae4edec05e775ca9632cf02d29c2a089b563b0d05cdf95aeca507de674553f28b4eadaca82d5549a86058f9996b07768686a5b02cb240dd9f1022100f4a63f5549e817547dca97b5c658038e8593cb78c5aba3c4642cc4cd031d868f022100d598d870ffe4a34df8de57047a50b97b71f4d23e323f527837c9edae88c7948302210098560c89a70385c36eb07fd7083235c4c1184e525d838aedf7128958bedfdbb1022051c0dab7057a8176ca966f3feb81123d4974a733df0f958525f547dfd1c271f90220446c2cafad455a671a8cf398e642e1be3b18a3d3aec2e67a9478f83c964c4f1f')
cert = binascii.unhexlify(b'308201d53082017f020203e8300d06092a864886f70d01010505003075310b30090603550406130258583114301206035504080c0b54686550726f76696e63653110300e06035504070c075468654369747931133011060355040a0c0a436f6d70616e7958595a31133011060355040b0c0a436f6d70616e7958595a3114301206035504030c0b546865486f73744e616d65301e170d3139313231383033333935355a170d3239313231353033333935355a3075310b30090603550406130258583114301206035504080c0b54686550726f76696e63653110300e06035504070c075468654369747931133011060355040a0c0a436f6d70616e7958595a31133011060355040b0c0a436f6d70616e7958595a3114301206035504030c0b546865486f73744e616d65305c300d06092a864886f70d0101010500034b003048024100cc20643fd3d9c21a0acba4f48f61aadd675f52175a9dcf07fbef610a6a6ba14abb891745cd18a1d4c056580d8ff1a639460f867013c8391cdc9f2e573b0f872d0203010001300d06092a864886f70d0101050500034100b0513fe2829e9ecbe55b6dd14c0ede7502bde5d46153c8e960ae3ebc247371b525caeb41bbcf34686015a44c50d226e66aef0a97a63874ca5944ef979b57f0b3')

class _Pipe(io.IOBase):

    def __init__(self):
        if False:
            return 10
        self._other = None
        self.block_reads = False
        self.block_writes = False
        self.write_buffers = []
        self.last_poll_arg = None

    def readinto(self, buf):
        if False:
            i = 10
            return i + 15
        if self.block_reads or len(self._other.write_buffers) == 0:
            return None
        read_buf = self._other.write_buffers[0]
        l = min(len(buf), len(read_buf))
        buf[:l] = read_buf[:l]
        if l == len(read_buf):
            self._other.write_buffers.pop(0)
        else:
            self._other.write_buffers[0] = read_buf[l:]
        return l

    def write(self, buf):
        if False:
            i = 10
            return i + 15
        if self.block_writes:
            return None
        self.write_buffers.append(memoryview(bytes(buf)))
        return len(buf)

    def ioctl(self, request, arg):
        if False:
            print('Hello World!')
        if request == _MP_STREAM_POLL:
            self.last_poll_arg = arg
            ret = 0
            if arg & _MP_STREAM_POLL_RD:
                if not self.block_reads and self._other.write_buffers:
                    ret |= _MP_STREAM_POLL_RD
            if arg & _MP_STREAM_POLL_WR:
                if not self.block_writes:
                    ret |= _MP_STREAM_POLL_WR
            return ret
        elif request == _MP_STREAM_CLOSE:
            return 0
        raise NotImplementedError()

    @classmethod
    def new_pair(cls):
        if False:
            for i in range(10):
                print('nop')
        p1 = cls()
        p2 = cls()
        p1._other = p2
        p2._other = p1
        return (p1, p2)

def assert_poll(s, i, arg, expected_arg, expected_ret):
    if False:
        return 10
    ret = s.ioctl(_MP_STREAM_POLL, arg)
    assert i.last_poll_arg == expected_arg
    i.last_poll_arg = None
    assert ret == expected_ret

def assert_raises(cb, *args, **kwargs):
    if False:
        print('Hello World!')
    try:
        cb(*args, **kwargs)
        raise AssertionError('should have raised')
    except Exception as exc:
        pass
(client_io, server_io) = _Pipe.new_pair()
client_io.block_reads = True
client_io.block_writes = True
client_sock = ssl.wrap_socket(client_io, do_handshake=False)
server_sock = ssl.wrap_socket(server_io, key=key, cert=cert, server_side=True, do_handshake=False)
assert client_sock.read(128) is None
assert_poll(client_sock, client_io, _MP_STREAM_POLL_RD, _MP_STREAM_POLL_WR, 0)
assert_poll(client_sock, client_io, _MP_STREAM_POLL_WR, _MP_STREAM_POLL_WR, 0)
client_io.block_writes = False
assert client_sock.read(128) is None
assert len(client_io.write_buffers) == 1
assert_poll(client_sock, client_io, _MP_STREAM_POLL_RD, _MP_STREAM_POLL_RD, 0)
assert_poll(client_sock, client_io, _MP_STREAM_POLL_WR, _MP_STREAM_POLL_WR, _MP_STREAM_POLL_WR)
client_sock.write(b'foo')
assert_poll(client_sock, client_io, _MP_STREAM_POLL_RD, _MP_STREAM_POLL_RD, 0)
assert_poll(client_sock, client_io, _MP_STREAM_POLL_WR, _MP_STREAM_POLL_RD, 0)
client_io.block_reads = False
while server_io.write_buffers or client_io.write_buffers:
    if server_io.write_buffers:
        assert client_sock.read(128) is None
    if client_io.write_buffers:
        assert server_sock.read(128) is None
client_sock.write(b'foo')
assert server_sock.read(3) == b'foo'
client_sock.write(b'foobar')
assert server_sock.read(3) == b'foo'
server_io.block_reads = True
assert_poll(server_sock, server_io, _MP_STREAM_POLL_RD, None, _MP_STREAM_POLL_RD)
assert server_sock.read(3) == b'bar'
(client_io, _) = _Pipe.new_pair()
client_sock = ssl.wrap_socket(client_io, do_handshake=False)
client_sock.close()
assert_poll(client_sock, client_io, _MP_STREAM_POLL_RD, None, _MP_STREAM_POLL_NVAL)
(client_io, server_io) = _Pipe.new_pair()
client_sock = ssl.wrap_socket(client_io, do_handshake=False)
server_io.write(b'fooba')
assert_poll(client_sock, client_io, _MP_STREAM_POLL_RD, _MP_STREAM_POLL_RD, _MP_STREAM_POLL_RD)
assert_raises(client_sock.read, 128)
assert_poll(client_sock, client_io, _MP_STREAM_POLL_RD, None, _MP_STREAM_POLL_NVAL)
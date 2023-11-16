import ssl
from contextlib import contextmanager
client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
client_ctx.check_hostname = False
client_ctx.verify_mode = ssl.CERT_NONE
cinb = ssl.MemoryBIO()
coutb = ssl.MemoryBIO()
cso = client_ctx.wrap_bio(cinb, coutb)
server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
server_ctx.load_cert_chain('server.crt', 'server.key', 'xxxx')
sinb = ssl.MemoryBIO()
soutb = ssl.MemoryBIO()
sso = server_ctx.wrap_bio(sinb, soutb, server_side=True)

@contextmanager
def expect(etype):
    if False:
        while True:
            i = 10
    try:
        yield
    except etype:
        pass
    else:
        raise AssertionError(f'expected {etype}')
with expect(ssl.SSLWantReadError):
    cso.do_handshake()
assert not cinb.pending
assert coutb.pending
with expect(ssl.SSLWantReadError):
    sso.do_handshake()
assert not sinb.pending
assert not soutb.pending
sinb.write(coutb.read())
with expect(ssl.SSLWantReadError):
    sso.do_handshake()
assert soutb.pending
cinb.write(soutb.read())
with expect(ssl.SSLWantReadError):
    cso.do_handshake()
sinb.write(coutb.read())
sso.do_handshake()
assert soutb.pending
cinb.write(soutb.read())
cso.do_handshake()
cso.write(b'hello')
sinb.write(coutb.read())
assert sso.read(10) == b'hello'
with expect(ssl.SSLWantReadError):
    sso.read(10)
assert not coutb.pending
assert not cinb.pending
sso.do_handshake()
assert not coutb.pending
assert not cinb.pending
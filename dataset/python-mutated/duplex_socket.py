from __future__ import annotations
import logging
import socket
import ssl
from asyncio.selector_events import BaseSelectorEventLoop
from localstack.utils.asyncio import run_sync
LOG = logging.getLogger(__name__)

class DuplexSocket(ssl.SSLSocket):
    """Simple duplex socket wrapper that allows serving HTTP/HTTPS over the same port."""

    def accept(self):
        if False:
            while True:
                i = 10
        (newsock, addr) = socket.socket.accept(self)
        if DuplexSocket.is_ssl_socket(newsock) is not False:
            newsock = self.context.wrap_socket(newsock, do_handshake_on_connect=self.do_handshake_on_connect, suppress_ragged_eofs=self.suppress_ragged_eofs, server_side=True)
        return (newsock, addr)

    @staticmethod
    def is_ssl_socket(newsock):
        if False:
            while True:
                i = 10
        'Returns True/False if the socket uses SSL or not, or None if the status cannot be\n        determined'

        def peek_ssl_header():
            if False:
                for i in range(10):
                    print('nop')
            peek_bytes = 5
            first_bytes = newsock.recv(peek_bytes, socket.MSG_PEEK)
            if len(first_bytes or '') != peek_bytes:
                return
            first_byte = first_bytes[0]
            return first_byte < 32 or first_byte >= 127
        try:
            return peek_ssl_header()
        except Exception:
            newsock.setblocking(1)
            newsock.settimeout(1)
            try:
                return peek_ssl_header()
            except Exception:
                return False

def enable_duplex_socket():
    if False:
        i = 10
        return i + 15
    '\n    Function which replaces the ssl.SSLContext.sslsocket_class with the DuplexSocket, enabling serving both,\n    HTTP and HTTPS connections on a single port.\n    '
    ssl.SSLContext.sslsocket_class = DuplexSocket

    async def _accept_connection2(self, protocol_factory, conn, extra, sslcontext, *args, **kwargs):
        is_ssl_socket = await run_sync(DuplexSocket.is_ssl_socket, conn)
        if is_ssl_socket is False:
            sslcontext = None
        result = await _accept_connection2_orig(self, protocol_factory, conn, extra, sslcontext, *args, **kwargs)
        return result
    if hasattr(BaseSelectorEventLoop, '_accept_connection2') and (not hasattr(BaseSelectorEventLoop, '_ls_patched')):
        _accept_connection2_orig = BaseSelectorEventLoop._accept_connection2
        BaseSelectorEventLoop._accept_connection2 = _accept_connection2
        BaseSelectorEventLoop._ls_patched = True
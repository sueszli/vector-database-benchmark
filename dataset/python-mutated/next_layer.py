"""
This addon determines the next protocol layer in our proxy stack.
Whenever a protocol layer in the proxy wants to pass a connection to a child layer and isn't sure which protocol comes
next, it calls the `next_layer` hook, which ends up here.
For example, if mitmproxy runs as a regular proxy, we first need to determine if
new clients start with a TLS handshake right away (Secure Web Proxy) or send a plaintext HTTP CONNECT request.
This addon here peeks at the incoming bytes and then makes a decision based on proxy mode, mitmproxy options, etc.

For a typical HTTPS request, this addon is called a couple of times: First to determine that we start with an HTTP layer
which processes the `CONNECT` request, a second time to determine that the client then starts negotiating TLS, and a
third time when we check if the protocol within that TLS stream is actually HTTP or something else.

Sometimes it's useful to hardcode specific logic in next_layer when one wants to do fancy things.
In that case it's not necessary to modify mitmproxy's source, adding a custom addon with a next_layer event hook
that sets nextlayer.layer works just as well.
"""
from __future__ import annotations
import logging
import re
import struct
import sys
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any
from typing import cast
from mitmproxy import ctx
from mitmproxy import dns
from mitmproxy import exceptions
from mitmproxy.net.tls import starts_like_dtls_record
from mitmproxy.net.tls import starts_like_tls_record
from mitmproxy.proxy import layer
from mitmproxy.proxy import layers
from mitmproxy.proxy import mode_specs
from mitmproxy.proxy import tunnel
from mitmproxy.proxy.context import Context
from mitmproxy.proxy.layer import Layer
from mitmproxy.proxy.layers import ClientQuicLayer
from mitmproxy.proxy.layers import ClientTLSLayer
from mitmproxy.proxy.layers import DNSLayer
from mitmproxy.proxy.layers import HttpLayer
from mitmproxy.proxy.layers import modes
from mitmproxy.proxy.layers import RawQuicLayer
from mitmproxy.proxy.layers import ServerQuicLayer
from mitmproxy.proxy.layers import ServerTLSLayer
from mitmproxy.proxy.layers import TCPLayer
from mitmproxy.proxy.layers import UDPLayer
from mitmproxy.proxy.layers.http import HTTPMode
from mitmproxy.proxy.layers.quic import quic_parse_client_hello
from mitmproxy.proxy.layers.tls import dtls_parse_client_hello
from mitmproxy.proxy.layers.tls import HTTP_ALPNS
from mitmproxy.proxy.layers.tls import parse_client_hello
from mitmproxy.tls import ClientHello
if sys.version_info < (3, 11):
    from typing_extensions import assert_never
else:
    from typing import assert_never
logger = logging.getLogger(__name__)

def stack_match(context: Context, layers: Sequence[type[Layer] | tuple[type[Layer], ...]]) -> bool:
    if False:
        return 10
    if len(context.layers) != len(layers):
        return False
    return all((expected is Any or isinstance(actual, expected) for (actual, expected) in zip(context.layers, layers)))

class NeedsMoreData(Exception):
    """Signal that the decision on which layer to put next needs to be deferred within the NextLayer addon."""

class NextLayer:
    ignore_hosts: Sequence[re.Pattern] = ()
    allow_hosts: Sequence[re.Pattern] = ()
    tcp_hosts: Sequence[re.Pattern] = ()
    udp_hosts: Sequence[re.Pattern] = ()

    def configure(self, updated):
        if False:
            return 10
        if 'tcp_hosts' in updated:
            self.tcp_hosts = [re.compile(x, re.IGNORECASE) for x in ctx.options.tcp_hosts]
        if 'udp_hosts' in updated:
            self.udp_hosts = [re.compile(x, re.IGNORECASE) for x in ctx.options.udp_hosts]
        if 'allow_hosts' in updated or 'ignore_hosts' in updated:
            if ctx.options.allow_hosts and ctx.options.ignore_hosts:
                raise exceptions.OptionsError('The allow_hosts and ignore_hosts options are mutually exclusive.')
            self.ignore_hosts = [re.compile(x, re.IGNORECASE) for x in ctx.options.ignore_hosts]
            self.allow_hosts = [re.compile(x, re.IGNORECASE) for x in ctx.options.allow_hosts]

    def next_layer(self, nextlayer: layer.NextLayer):
        if False:
            return 10
        if nextlayer.layer:
            return
        try:
            nextlayer.layer = self._next_layer(nextlayer.context, nextlayer.data_client(), nextlayer.data_server())
        except NeedsMoreData:
            logger.info(f'Deferring layer decision, not enough data: {nextlayer.data_client().hex()}')

    def _next_layer(self, context: Context, data_client: bytes, data_server: bytes) -> Layer | None:
        if False:
            while True:
                i = 10
        assert context.layers

        def s(*layers):
            if False:
                return 10
            return stack_match(context, layers)
        tcp_based = context.client.transport_protocol == 'tcp'
        udp_based = context.client.transport_protocol == 'udp'
        if self._ignore_connection(context, data_client):
            return layers.TCPLayer(context, ignore=True) if tcp_based else layers.UDPLayer(context, ignore=True)
        if s(modes.ReverseProxy):
            return self._setup_reverse_proxy(context, data_client)
        if s((modes.HttpProxy, modes.HttpUpstreamProxy)):
            return self._setup_explicit_http_proxy(context, data_client)
        is_tls_or_dtls = tcp_based and starts_like_tls_record(data_client) or (udp_based and starts_like_dtls_record(data_client))
        if is_tls_or_dtls:
            server_tls = ServerTLSLayer(context)
            server_tls.child_layer = ClientTLSLayer(context)
            return server_tls
        if udp_based and _starts_like_quic(data_client):
            server_quic = ServerQuicLayer(context)
            server_quic.child_layer = ClientQuicLayer(context)
            return server_quic
        if tcp_based and self._is_destination_in_hosts(context, self.tcp_hosts):
            return layers.TCPLayer(context)
        if udp_based and self._is_destination_in_hosts(context, self.udp_hosts):
            return layers.UDPLayer(context)
        if udp_based:
            try:
                dns.Message.unpack(data_client)
            except struct.error:
                pass
            else:
                return layers.DNSLayer(context)
        if udp_based:
            return layers.UDPLayer(context)
        very_likely_http = context.client.alpn and context.client.alpn in HTTP_ALPNS
        probably_no_http = not very_likely_http and (len(data_client) < 3 or not data_client[:3].isalpha() or data_server)
        if ctx.options.rawtcp and probably_no_http:
            return layers.TCPLayer(context)
        return layers.HttpLayer(context, HTTPMode.transparent)

    def _ignore_connection(self, context: Context, data_client: bytes) -> bool | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns:\n            True, if the connection should be ignored.\n            False, if it should not be ignored.\n\n        Raises:\n            NeedsMoreData, if we need to wait for more input data.\n        '
        if not ctx.options.ignore_hosts and (not ctx.options.allow_hosts):
            return False
        hostnames: list[str] = []
        if context.server.peername and (peername := context.server.peername[0]):
            hostnames.append(peername)
        if context.server.address and (server_address := context.server.address[0]):
            hostnames.append(server_address)
        if (client_hello := self._get_client_hello(context, data_client)) and client_hello.sni:
            hostnames.append(client_hello.sni)
        if not hostnames:
            return False
        if ctx.options.ignore_hosts:
            return any((re.search(rex, host, re.IGNORECASE) for host in hostnames for rex in ctx.options.ignore_hosts))
        elif ctx.options.allow_hosts:
            return not any((re.search(rex, host, re.IGNORECASE) for host in hostnames for rex in ctx.options.allow_hosts))
        else:
            raise AssertionError()

    def _get_client_hello(self, context: Context, data_client: bytes) -> ClientHello | None:
        if False:
            return 10
        '\n        Try to read a TLS/DTLS/QUIC ClientHello from data_client.\n\n        Returns:\n            A complete ClientHello, or None, if no ClientHello was found.\n\n        Raises:\n            NeedsMoreData, if the ClientHello is incomplete.\n        '
        match context.client.transport_protocol:
            case 'tcp':
                if starts_like_tls_record(data_client):
                    try:
                        ch = parse_client_hello(data_client)
                    except ValueError:
                        pass
                    else:
                        if ch is None:
                            raise NeedsMoreData
                        return ch
                return None
            case 'udp':
                try:
                    return quic_parse_client_hello(data_client)
                except ValueError:
                    pass
                try:
                    ch = dtls_parse_client_hello(data_client)
                except ValueError:
                    pass
                else:
                    if ch is None:
                        raise NeedsMoreData
                    return ch
                return None
            case _:
                assert_never(context.client.transport_protocol)

    def _setup_reverse_proxy(self, context: Context, data_client: bytes) -> Layer:
        if False:
            while True:
                i = 10
        spec = cast(mode_specs.ReverseMode, context.client.proxy_mode)
        stack = tunnel.LayerStack()
        match spec.scheme:
            case 'http':
                if starts_like_tls_record(data_client):
                    stack /= ClientTLSLayer(context)
                stack /= HttpLayer(context, HTTPMode.transparent)
            case 'https':
                stack /= ServerTLSLayer(context)
                if starts_like_tls_record(data_client):
                    stack /= ClientTLSLayer(context)
                stack /= HttpLayer(context, HTTPMode.transparent)
            case 'tcp':
                if starts_like_tls_record(data_client):
                    stack /= ClientTLSLayer(context)
                stack /= TCPLayer(context)
            case 'tls':
                stack /= ServerTLSLayer(context)
                if starts_like_tls_record(data_client):
                    stack /= ClientTLSLayer(context)
                stack /= TCPLayer(context)
            case 'udp':
                if starts_like_dtls_record(data_client):
                    stack /= ClientTLSLayer(context)
                stack /= UDPLayer(context)
            case 'dtls':
                stack /= ServerTLSLayer(context)
                if starts_like_dtls_record(data_client):
                    stack /= ClientTLSLayer(context)
                stack /= UDPLayer(context)
            case 'dns':
                stack /= DNSLayer(context)
            case 'http3':
                stack /= ServerQuicLayer(context)
                stack /= ClientQuicLayer(context)
                stack /= HttpLayer(context, HTTPMode.transparent)
            case 'quic':
                stack /= ServerQuicLayer(context)
                stack /= ClientQuicLayer(context)
                stack /= RawQuicLayer(context)
            case _:
                assert_never(spec.scheme)
        return stack[0]

    def _setup_explicit_http_proxy(self, context: Context, data_client: bytes) -> Layer:
        if False:
            i = 10
            return i + 15
        stack = tunnel.LayerStack()
        if context.client.transport_protocol == 'udp':
            stack /= layers.ClientQuicLayer(context)
        elif starts_like_tls_record(data_client):
            stack /= layers.ClientTLSLayer(context)
        if isinstance(context.layers[0], modes.HttpUpstreamProxy):
            stack /= layers.HttpLayer(context, HTTPMode.upstream)
        else:
            stack /= layers.HttpLayer(context, HTTPMode.regular)
        return stack[0]

    def _is_destination_in_hosts(self, context: Context, hosts: Iterable[re.Pattern]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return any((context.server.address and rex.search(context.server.address[0]) or (context.client.sni and rex.search(context.client.sni)) for rex in hosts))

def _starts_like_quic(data_client: bytes) -> bool:
    if False:
        i = 10
        return i + 15
    try:
        quic_parse_client_hello(data_client)
    except ValueError:
        return False
    else:
        return True
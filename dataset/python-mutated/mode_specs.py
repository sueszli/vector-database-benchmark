"""
This module is responsible for parsing proxy mode specifications such as
`"regular"`, `"reverse:https://example.com"`, or `"socks5@1234"`. The general syntax is

    mode [: mode_configuration] [@ [listen_addr:]listen_port]

For a full example, consider `reverse:https://example.com@127.0.0.1:443`.
This would spawn a reverse proxy on port 443 bound to localhost.
The mode is `reverse`, and the mode data is `https://example.com`.
Examples:

    mode = ProxyMode.parse("regular@1234")
    assert mode.listen_port == 1234
    assert isinstance(mode, RegularMode)

    ProxyMode.parse("reverse:example.com@invalid-port")  # ValueError

    RegularMode.parse("regular")  # ok
    RegularMode.parse("socks5")  # ValueError

"""
from __future__ import annotations
import dataclasses
import sys
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import ClassVar
from typing import Literal
import mitmproxy_rs
from mitmproxy.coretypes.serializable import Serializable
from mitmproxy.net import server_spec
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

@dataclass(frozen=True)
class ProxyMode(Serializable, metaclass=ABCMeta):
    """
    Parsed representation of a proxy mode spec. Subclassed for each specific mode,
    which then does its own data validation.
    """
    full_spec: str
    'The full proxy mode spec as entered by the user.'
    data: str
    'The (raw) mode data, i.e. the part after the mode name.'
    custom_listen_host: str | None
    'A custom listen host, if specified in the spec.'
    custom_listen_port: int | None
    'A custom listen port, if specified in the spec.'
    type_name: ClassVar[str]
    'The unique name for this proxy mode, e.g. "regular" or "reverse".'
    __types: ClassVar[dict[str, type[ProxyMode]]] = {}

    def __init_subclass__(cls, **kwargs):
        if False:
            print('Hello World!')
        cls.type_name = cls.__name__.removesuffix('Mode').lower()
        assert cls.type_name not in ProxyMode.__types
        ProxyMode.__types[cls.type_name] = cls

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'ProxyMode.parse({self.full_spec!r})'

    @abstractmethod
    def __post_init__(self) -> None:
        if False:
            return 10
        'Validation of data happens here.'

    @property
    @abstractmethod
    def description(self) -> str:
        if False:
            print('Hello World!')
        'The mode description that will be used in server logs and UI.'

    @property
    def default_port(self) -> int:
        if False:
            return 10
        '\n        Default listen port of servers for this mode, see `ProxyMode.listen_port()`.\n        '
        return 8080

    @property
    @abstractmethod
    def transport_protocol(self) -> Literal['tcp', 'udp'] | None:
        if False:
            i = 10
            return i + 15
        "The transport protocol used by this mode's server."

    @classmethod
    @cache
    def parse(cls, spec: str) -> Self:
        if False:
            print('Hello World!')
        '\n        Parse a proxy mode specification and return the corresponding `ProxyMode` instance.\n        '
        (head, _, listen_at) = spec.rpartition('@')
        if not head:
            head = listen_at
            listen_at = ''
        (mode, _, data) = head.partition(':')
        if listen_at:
            if ':' in listen_at:
                (host, _, port_str) = listen_at.rpartition(':')
            else:
                host = None
                port_str = listen_at
            try:
                port = int(port_str)
                if port < 0 or 65535 < port:
                    raise ValueError
            except ValueError:
                raise ValueError(f'invalid port: {port_str}')
        else:
            host = None
            port = None
        try:
            mode_cls = ProxyMode.__types[mode.lower()]
        except KeyError:
            raise ValueError(f'unknown mode')
        if not issubclass(mode_cls, cls):
            raise ValueError(f'{mode!r} is not a spec for a {cls.type_name} mode')
        return mode_cls(full_spec=spec, data=data, custom_listen_host=host, custom_listen_port=port)

    def listen_host(self, default: str | None=None) -> str:
        if False:
            return 10
        '\n        Return the address a server for this mode should listen on. This can be either directly\n        specified in the spec or taken from a user-configured global default (`options.listen_host`).\n        By default, return an empty string to listen on all hosts.\n        '
        if self.custom_listen_host is not None:
            return self.custom_listen_host
        elif default is not None:
            return default
        else:
            return ''

    def listen_port(self, default: int | None=None) -> int:
        if False:
            print('Hello World!')
        '\n        Return the port a server for this mode should listen on. This can be either directly\n        specified in the spec, taken from a user-configured global default (`options.listen_port`),\n        or from `ProxyMode.default_port`.\n        '
        if self.custom_listen_port is not None:
            return self.custom_listen_port
        elif default is not None:
            return default
        else:
            return self.default_port

    @classmethod
    def from_state(cls, state):
        if False:
            while True:
                i = 10
        return ProxyMode.parse(state)

    def get_state(self):
        if False:
            while True:
                i = 10
        return self.full_spec

    def set_state(self, state):
        if False:
            return 10
        if state != self.full_spec:
            raise dataclasses.FrozenInstanceError('Proxy modes are immutable.')
TCP: Literal['tcp', 'udp'] = 'tcp'
UDP: Literal['tcp', 'udp'] = 'udp'

def _check_empty(data):
    if False:
        while True:
            i = 10
    if data:
        raise ValueError('mode takes no arguments')

class RegularMode(ProxyMode):
    """A regular HTTP(S) proxy that is interfaced with `HTTP CONNECT` calls (or absolute-form HTTP requests)."""
    description = 'HTTP(S) proxy'
    transport_protocol = TCP

    def __post_init__(self) -> None:
        if False:
            i = 10
            return i + 15
        _check_empty(self.data)

class TransparentMode(ProxyMode):
    """A transparent proxy, see https://docs.mitmproxy.org/dev/howto-transparent/"""
    description = 'Transparent Proxy'
    transport_protocol = TCP

    def __post_init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        _check_empty(self.data)

class UpstreamMode(ProxyMode):
    """A regular HTTP(S) proxy, but all connections are forwarded to a second upstream HTTP(S) proxy."""
    description = 'HTTP(S) proxy (upstream mode)'
    transport_protocol = TCP
    scheme: Literal['http', 'https']
    address: tuple[str, int]

    def __post_init__(self) -> None:
        if False:
            i = 10
            return i + 15
        (scheme, self.address) = server_spec.parse(self.data, default_scheme='http')
        if scheme != 'http' and scheme != 'https':
            raise ValueError('invalid upstream proxy scheme')
        self.scheme = scheme

class ReverseMode(ProxyMode):
    """A reverse proxy. This acts like a normal server, but redirects all requests to a fixed target."""
    description = 'reverse proxy'
    transport_protocol = TCP
    scheme: Literal['http', 'https', 'http3', 'tls', 'dtls', 'tcp', 'udp', 'dns', 'quic']
    address: tuple[str, int]

    def __post_init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (self.scheme, self.address) = server_spec.parse(self.data, default_scheme='https')
        if self.scheme in ('http3', 'dtls', 'udp', 'dns', 'quic'):
            self.transport_protocol = UDP
        self.description = f'{self.description} to {self.data}'

    @property
    def default_port(self) -> int:
        if False:
            while True:
                i = 10
        if self.scheme == 'dns':
            return 53
        return super().default_port

class Socks5Mode(ProxyMode):
    """A SOCKSv5 proxy."""
    description = 'SOCKS v5 proxy'
    default_port = 1080
    transport_protocol = TCP

    def __post_init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        _check_empty(self.data)

class DnsMode(ProxyMode):
    """A DNS server."""
    description = 'DNS server'
    default_port = 53
    transport_protocol = UDP

    def __post_init__(self) -> None:
        if False:
            print('Hello World!')
        _check_empty(self.data)

class WireGuardMode(ProxyMode):
    """Proxy Server based on WireGuard"""
    description = 'WireGuard server'
    default_port = 51820
    transport_protocol = UDP

    def __post_init__(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

class LocalMode(ProxyMode):
    """OS-level transparent proxy."""
    description = 'Local redirector'
    transport_protocol = None

    def __post_init__(self) -> None:
        if False:
            print('Hello World!')
        mitmproxy_rs.LocalRedirector.describe_spec(self.data)

class OsProxyMode(ProxyMode):
    """Deprecated alias for LocalMode"""
    description = 'Deprecated alias for LocalMode'
    transport_protocol = None

    def __post_init__(self) -> None:
        if False:
            while True:
                i = 10
        raise ValueError('osproxy mode has been renamed to local mode. Thanks for trying our experimental features!')
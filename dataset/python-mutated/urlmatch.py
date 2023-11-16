"""A Chromium-like URL matching pattern.

See:
https://developer.chrome.com/apps/match_patterns
https://cs.chromium.org/chromium/src/extensions/common/url_pattern.cc
https://cs.chromium.org/chromium/src/extensions/common/url_pattern.h

Based on the following commit in Chromium:
https://chromium.googlesource.com/chromium/src/+/6f4a6681eae01c2036336c18b06303e16a304a7c
(October 10 2020, newest commit as per October 28th 2020)
"""
import ipaddress
import fnmatch
import urllib.parse
from typing import Any, Optional, Tuple
from qutebrowser.qt.core import QUrl
from qutebrowser.utils import utils, qtutils

class ParseError(Exception):
    """Raised when a pattern could not be parsed."""

class UrlPattern:
    """A Chromium-like URL matching pattern.

    Class attributes:
        _DEFAULT_PORTS: The default ports used for schemes which support ports.
        _SCHEMES_WITHOUT_HOST: Schemes which don't need a host.

    Attributes:
        host: The host to match to, or None for any host.
        _pattern: The given pattern as string.
        _match_all: Whether the pattern should match all URLs.
        _match_subdomains: Whether the pattern should match subdomains of the
                           given host.
        _scheme: The scheme to match to, or None to match any scheme.
                 Note that with Chromium, '*'/None only matches http/https and
                 not file/ftp. We deviate from that as per-URL settings aren't
                 security relevant.
        _path: The path to match to, or None for any path.
        _port: The port to match to as integer, or None for any port.
    """
    _DEFAULT_PORTS = {'https': 443, 'http': 80, 'ftp': 21}
    _SCHEMES_WITHOUT_HOST = ['about', 'file', 'data', 'javascript']

    def __init__(self, pattern: str) -> None:
        if False:
            print('Hello World!')
        self._pattern = pattern
        self._match_all = False
        self._match_subdomains: bool = False
        self._scheme: Optional[str] = None
        self.host: Optional[str] = None
        self._path: Optional[str] = None
        self._port: Optional[int] = None
        if pattern == '<all_urls>':
            self._match_all = True
            return
        if '\x00' in pattern:
            raise ParseError('May not contain NUL byte')
        pattern = self._fixup_pattern(pattern)
        try:
            parsed = urllib.parse.urlparse(pattern)
        except ValueError as e:
            raise ParseError(str(e))
        assert parsed is not None
        self._init_scheme(parsed)
        self._init_host(parsed)
        self._init_path(parsed)
        self._init_port(parsed)

    def _to_tuple(self) -> Tuple[bool, bool, Optional[str], Optional[str], Optional[str], Optional[int]]:
        if False:
            print('Hello World!')
        'Get a pattern with information used for __eq__/__hash__.'
        return (self._match_all, self._match_subdomains, self._scheme, self.host, self._path, self._port)

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(self._to_tuple())

    def __eq__(self, other: Any) -> bool:
        if False:
            return 10
        if not isinstance(other, UrlPattern):
            return NotImplemented
        return self._to_tuple() == other._to_tuple()

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return utils.get_repr(self, pattern=self._pattern, constructor=True)

    def __str__(self) -> str:
        if False:
            return 10
        return self._pattern

    def _fixup_pattern(self, pattern: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Make sure the given pattern is parseable by urllib.parse.'
        if pattern.startswith('*:'):
            pattern = 'any:' + pattern[2:]
        schemes = tuple((s + ':' for s in self._SCHEMES_WITHOUT_HOST))
        if '://' not in pattern and (not pattern.startswith(schemes)):
            pattern = 'any://' + pattern
        if pattern.startswith('file://') and (not pattern.startswith('file:///')):
            pattern = 'file:///' + pattern[len('file://'):]
        return pattern

    def _init_scheme(self, parsed: urllib.parse.ParseResult) -> None:
        if False:
            while True:
                i = 10
        'Parse the scheme from the given URL.\n\n        Deviation from Chromium:\n        - We assume * when no scheme has been given.\n        '
        if not parsed.scheme:
            raise ParseError('Missing scheme')
        if parsed.scheme == 'any':
            self._scheme = None
            return
        self._scheme = parsed.scheme

    def _init_path(self, parsed: urllib.parse.ParseResult) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Parse the path from the given URL.\n\n        Deviation from Chromium:\n        - We assume * when no path has been given.\n        '
        if self._scheme == 'about' and (not parsed.path.strip()):
            raise ParseError('Pattern without path')
        if parsed.path == '/*':
            self._path = None
        elif not parsed.path:
            self._path = None
        else:
            self._path = parsed.path

    def _init_host(self, parsed: urllib.parse.ParseResult) -> None:
        if False:
            return 10
        "Parse the host from the given URL.\n\n        Deviation from Chromium:\n        - http://:1234/ is not a valid URL because it has no host.\n        - We don't allow patterns for dot/space hosts which QUrl considers\n          invalid.\n        "
        if parsed.hostname is None or not parsed.hostname.strip():
            if self._scheme not in self._SCHEMES_WITHOUT_HOST:
                raise ParseError('Pattern without host')
            assert self.host is None
            return
        if parsed.netloc.startswith('['):
            url = QUrl()
            url.setHost(parsed.hostname)
            if not url.isValid():
                raise ParseError(url.errorString())
            self.host = url.host()
            return
        if parsed.hostname == '*':
            self._match_subdomains = True
            hostname = None
        elif parsed.hostname.startswith('*.'):
            if len(parsed.hostname) == 2:
                raise ParseError('Pattern without host')
            self._match_subdomains = True
            hostname = parsed.hostname[2:]
        elif set(parsed.hostname) in {frozenset('.'), frozenset('. ')}:
            raise ParseError('Invalid host')
        else:
            hostname = parsed.hostname
        if hostname is None:
            self.host = None
        elif '*' in hostname:
            raise ParseError('Invalid host wildcard')
        else:
            self.host = hostname.rstrip('.')

    def _init_port(self, parsed: urllib.parse.ParseResult) -> None:
        if False:
            print('Hello World!')
        'Parse the port from the given URL.\n\n        Deviation from Chromium:\n        - We use None instead of "*" if there\'s no port filter.\n        '
        if parsed.netloc.endswith(':*'):
            self._port = None
        elif parsed.netloc.endswith(':'):
            raise ParseError('Invalid port: Port is empty')
        else:
            try:
                self._port = parsed.port
            except ValueError as e:
                raise ParseError('Invalid port: {}'.format(e))
        scheme_has_port = self._scheme in list(self._DEFAULT_PORTS) or self._scheme is None
        if self._port is not None and (not scheme_has_port):
            raise ParseError('Ports are unsupported with {} scheme'.format(self._scheme))

    def _matches_scheme(self, scheme: str) -> bool:
        if False:
            while True:
                i = 10
        return self._scheme is None or self._scheme == scheme

    def _matches_host(self, host: str) -> bool:
        if False:
            while True:
                i = 10
        host = host.rstrip('.')
        if self.host is None:
            return True
        if host == self.host:
            return True
        if not self._match_subdomains:
            return False
        if not utils.raises(ValueError, ipaddress.ip_address, host):
            return False
        if len(host) <= len(self.host) + 1:
            return False
        if not host.endswith(self.host):
            return False
        return host[len(host) - len(self.host) - 1] == '.'

    def _matches_port(self, scheme: str, port: int) -> bool:
        if False:
            while True:
                i = 10
        if port == -1 and scheme in self._DEFAULT_PORTS:
            port = self._DEFAULT_PORTS[scheme]
        return self._port is None or self._port == port

    def _matches_path(self, path: str) -> bool:
        if False:
            return 10
        'Match the URL\'s path.\n\n        Deviations from Chromium:\n        - Chromium only matches <all_urls> with "javascript:" (pathless); but\n          we also match *://*/* and friends.\n        '
        if self._path is None:
            return True
        if path + '/*' == self._path:
            return True
        return fnmatch.fnmatchcase(path, self._path)

    def matches(self, qurl: QUrl) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if the pattern matches the given QUrl.'
        qtutils.ensure_valid(qurl)
        if self._match_all:
            return True
        if not self._matches_scheme(qurl.scheme()):
            return False
        if not self._matches_host(qurl.host()):
            return False
        if not self._matches_port(qurl.scheme(), qurl.port()):
            return False
        if not self._matches_path(qurl.path()):
            return False
        return True
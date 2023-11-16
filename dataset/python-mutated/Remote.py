from contextlib import contextmanager
import http.client
import re
import socket
import sys
import xmlrpc.client
from datetime import date, datetime, timedelta
from xml.parsers.expat import ExpatError
from robot.errors import RemoteError
from robot.utils import DotDict, is_bytes, is_dict_like, is_list_like, is_number, is_string, safe_str, timestr_to_secs

class Remote:
    ROBOT_LIBRARY_SCOPE = 'TEST SUITE'

    def __init__(self, uri='http://127.0.0.1:8270', timeout=None):
        if False:
            return 10
        'Connects to a remote server at ``uri``.\n\n        Optional ``timeout`` can be used to specify a timeout to wait when\n        initially connecting to the server and if a connection accidentally\n        closes. Timeout can be given as seconds (e.g. ``60``) or using\n        Robot Framework time format (e.g. ``60s``, ``2 minutes 10 seconds``).\n\n        The default timeout is typically several minutes, but it depends on\n        the operating system and its configuration. Notice that setting\n        a timeout that is shorter than keyword execution time will interrupt\n        the keyword.\n        '
        if '://' not in uri:
            uri = 'http://' + uri
        if timeout:
            timeout = timestr_to_secs(timeout)
        self._uri = uri
        self._client = XmlRpcRemoteClient(uri, timeout)
        self._lib_info = None
        self._lib_info_initialized = False

    def get_keyword_names(self):
        if False:
            while True:
                i = 10
        if self._is_lib_info_available():
            return [name for name in self._lib_info if not (name[:2] == '__' and name[-2:] == '__')]
        try:
            return self._client.get_keyword_names()
        except TypeError as error:
            raise RuntimeError(f'Connecting remote server at {self._uri} failed: {error}')

    def _is_lib_info_available(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._lib_info_initialized:
            try:
                self._lib_info = self._client.get_library_information()
            except TypeError:
                pass
            self._lib_info_initialized = True
        return self._lib_info is not None

    def get_keyword_arguments(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self._get_kw_info(name, 'args', self._client.get_keyword_arguments, default=['*args'])

    def _get_kw_info(self, kw, info, getter, default=None):
        if False:
            print('Hello World!')
        if self._is_lib_info_available():
            return self._lib_info[kw].get(info, default)
        try:
            return getter(kw)
        except TypeError:
            return default

    def get_keyword_types(self, name):
        if False:
            while True:
                i = 10
        return self._get_kw_info(name, 'types', self._client.get_keyword_types, default=())

    def get_keyword_tags(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self._get_kw_info(name, 'tags', self._client.get_keyword_tags)

    def get_keyword_documentation(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self._get_kw_info(name, 'doc', self._client.get_keyword_documentation)

    def run_keyword(self, name, args, kwargs):
        if False:
            return 10
        coercer = ArgumentCoercer()
        args = coercer.coerce(args)
        kwargs = coercer.coerce(kwargs)
        result = RemoteResult(self._client.run_keyword(name, args, kwargs))
        sys.stdout.write(result.output)
        if result.status != 'PASS':
            raise RemoteError(result.error, result.traceback, result.fatal, result.continuable)
        return result.return_

class ArgumentCoercer:
    binary = re.compile('[\x00-\x08\x0b\x0c\x0e-\x1f]')

    def coerce(self, argument):
        if False:
            for i in range(10):
                print('nop')
        for (handles, handler) in [(is_string, self._handle_string), (self._no_conversion_needed, self._pass_through), (self._is_date, self._handle_date), (self._is_timedelta, self._handle_timedelta), (is_dict_like, self._coerce_dict), (is_list_like, self._coerce_list)]:
            if handles(argument):
                return handler(argument)
        return self._to_string(argument)

    def _no_conversion_needed(self, arg):
        if False:
            for i in range(10):
                print('nop')
        return is_number(arg) or is_bytes(arg) or isinstance(arg, datetime)

    def _handle_string(self, arg):
        if False:
            return 10
        if self.binary.search(arg):
            return self._handle_binary_in_string(arg)
        return arg

    def _handle_binary_in_string(self, arg):
        if False:
            return 10
        try:
            return arg.encode('latin-1')
        except UnicodeError:
            raise ValueError(f'Cannot represent {arg!r} as binary.')

    def _pass_through(self, arg):
        if False:
            for i in range(10):
                print('nop')
        return arg

    def _is_date(self, arg):
        if False:
            i = 10
            return i + 15
        return isinstance(arg, date)

    def _handle_date(self, arg):
        if False:
            i = 10
            return i + 15
        return datetime(arg.year, arg.month, arg.day)

    def _is_timedelta(self, arg):
        if False:
            print('Hello World!')
        return isinstance(arg, timedelta)

    def _handle_timedelta(self, arg):
        if False:
            i = 10
            return i + 15
        return arg.total_seconds()

    def _coerce_list(self, arg):
        if False:
            for i in range(10):
                print('nop')
        return [self.coerce(item) for item in arg]

    def _coerce_dict(self, arg):
        if False:
            while True:
                i = 10
        return {self._to_key(key): self.coerce(arg[key]) for key in arg}

    def _to_key(self, item):
        if False:
            print('Hello World!')
        item = self._to_string(item)
        self._validate_key(item)
        return item

    def _to_string(self, item):
        if False:
            i = 10
            return i + 15
        item = safe_str(item) if item is not None else ''
        return self._handle_string(item)

    def _validate_key(self, key):
        if False:
            i = 10
            return i + 15
        if isinstance(key, bytes):
            raise ValueError(f'Dictionary keys cannot be binary. Got {key!r}.')

class RemoteResult:

    def __init__(self, result):
        if False:
            for i in range(10):
                print('nop')
        if not (is_dict_like(result) and 'status' in result):
            raise RuntimeError(f'Invalid remote result dictionary: {result!r}')
        self.status = result['status']
        self.output = safe_str(self._get(result, 'output'))
        self.return_ = self._get(result, 'return')
        self.error = safe_str(self._get(result, 'error'))
        self.traceback = safe_str(self._get(result, 'traceback'))
        self.fatal = bool(self._get(result, 'fatal', False))
        self.continuable = bool(self._get(result, 'continuable', False))

    def _get(self, result, key, default=''):
        if False:
            for i in range(10):
                print('nop')
        value = result.get(key, default)
        return self._convert(value)

    def _convert(self, value):
        if False:
            while True:
                i = 10
        if is_dict_like(value):
            return DotDict(((k, self._convert(v)) for (k, v) in value.items()))
        if is_list_like(value):
            return [self._convert(v) for v in value]
        return value

class XmlRpcRemoteClient:

    def __init__(self, uri, timeout=None):
        if False:
            while True:
                i = 10
        self.uri = uri
        self.timeout = timeout

    @property
    @contextmanager
    def _server(self):
        if False:
            for i in range(10):
                print('nop')
        if self.uri.startswith('https://'):
            transport = TimeoutHTTPSTransport(timeout=self.timeout)
        else:
            transport = TimeoutHTTPTransport(timeout=self.timeout)
        server = xmlrpc.client.ServerProxy(self.uri, encoding='UTF-8', use_builtin_types=True, transport=transport)
        try:
            yield server
        except (socket.error, xmlrpc.client.Error) as err:
            raise TypeError(err)
        finally:
            server('close')()

    def get_library_information(self):
        if False:
            for i in range(10):
                print('nop')
        with self._server as server:
            return server.get_library_information()

    def get_keyword_names(self):
        if False:
            i = 10
            return i + 15
        with self._server as server:
            return server.get_keyword_names()

    def get_keyword_arguments(self, name):
        if False:
            print('Hello World!')
        with self._server as server:
            return server.get_keyword_arguments(name)

    def get_keyword_types(self, name):
        if False:
            return 10
        with self._server as server:
            return server.get_keyword_types(name)

    def get_keyword_tags(self, name):
        if False:
            print('Hello World!')
        with self._server as server:
            return server.get_keyword_tags(name)

    def get_keyword_documentation(self, name):
        if False:
            print('Hello World!')
        with self._server as server:
            return server.get_keyword_documentation(name)

    def run_keyword(self, name, args, kwargs):
        if False:
            return 10
        with self._server as server:
            run_keyword_args = [name, args, kwargs] if kwargs else [name, args]
            try:
                return server.run_keyword(*run_keyword_args)
            except xmlrpc.client.Fault as err:
                message = err.faultString
            except socket.error as err:
                message = f'Connection to remote server broken: {err}'
            except ExpatError as err:
                message = f'Processing XML-RPC return value failed. Most often this happens when the return value contains characters that are not valid in XML. Original error was: ExpatError: {err}'
            raise RuntimeError(message)

class TimeoutHTTPTransport(xmlrpc.client.Transport):
    _connection_class = http.client.HTTPConnection

    def __init__(self, timeout=None):
        if False:
            i = 10
            return i + 15
        super().__init__(use_builtin_types=True)
        self.timeout = timeout or socket._GLOBAL_DEFAULT_TIMEOUT

    def make_connection(self, host):
        if False:
            return 10
        if self._connection and host == self._connection[0]:
            return self._connection[1]
        (chost, self._extra_headers, x509) = self.get_host_info(host)
        self._connection = (host, self._connection_class(chost, timeout=self.timeout))
        return self._connection[1]

class TimeoutHTTPSTransport(TimeoutHTTPTransport):
    _connection_class = http.client.HTTPSConnection
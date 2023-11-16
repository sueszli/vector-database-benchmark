"""Evaluation of PAC scripts."""
import sys
import functools
from typing import Optional, cast
from qutebrowser.qt import machinery
from qutebrowser.qt.core import QObject, pyqtSignal, pyqtSlot, QUrl
from qutebrowser.qt.network import QNetworkProxy, QNetworkRequest, QHostInfo, QNetworkReply, QNetworkAccessManager, QHostAddress
from qutebrowser.qt.qml import QJSEngine, QJSValue
from qutebrowser.utils import log, qtlog, utils, qtutils, resources, urlutils

class ParseProxyError(Exception):
    """Error while parsing PAC result string."""

class EvalProxyError(Exception):
    """Error while evaluating PAC script."""

def _js_slot(*args):
    if False:
        for i in range(10):
            print('nop')
    'Wrap a methods as a JavaScript function.\n\n    Register a PACContext method as a JavaScript function, and catch\n    exceptions returning them as JavaScript Error objects.\n\n    Args:\n        args: Types of method arguments.\n\n    Return: Wrapped method.\n    '

    def _decorator(method):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(method)
        def new_method(self, *args, **kwargs):
            if False:
                print('Hello World!')
            'Call the underlying function.'
            try:
                return method(self, *args, **kwargs)
            except:
                e = str(sys.exc_info()[0])
                log.network.exception('PAC evaluation error')
                return self._error_con.callAsConstructor([e])
        deco = pyqtSlot(*args, result='QJSValue')
        return deco(new_method)
    return _decorator

class _PACContext(QObject):
    """Implementation of PAC API functions that require native calls.

    See https://developer.mozilla.org/en-US/docs/Mozilla/Projects/Necko/Proxy_Auto-Configuration_(PAC)_file
    """
    JS_DEFINITIONS = '\n        function dnsResolve(host) {\n            return PAC.dnsResolve(host);\n        }\n\n        function myIpAddress() {\n            return PAC.myIpAddress();\n        }\n    '

    def __init__(self, engine):
        if False:
            return 10
        'Create a new PAC API implementation instance.\n\n        Args:\n            engine: QJSEngine which is used for running PAC.\n        '
        super().__init__(parent=engine)
        self._engine = engine
        self._error_con = engine.globalObject().property('Error')

    @_js_slot(str)
    def dnsResolve(self, host):
        if False:
            while True:
                i = 10
        'Resolve a DNS hostname.\n\n        Resolves the given DNS hostname into an IP address, and returns it\n        in the dot-separated format as a string.\n\n        Args:\n            host: hostname to resolve.\n        '
        ips = QHostInfo.fromName(host)
        if ips.error() != QHostInfo.HostInfoError.NoError or not ips.addresses():
            err_f = 'Failed to resolve host during PAC evaluation: {}'
            log.network.info(err_f.format(host))
            return QJSValue(QJSValue.SpecialValue.NullValue)
        else:
            return ips.addresses()[0].toString()

    @_js_slot()
    def myIpAddress(self):
        if False:
            return 10
        'Get host IP address.\n\n        Return the server IP address of the current machine, as a string in\n        the dot-separated integer format.\n        '
        return QHostAddress(QHostAddress.SpecialAddress.LocalHost).toString()

class PACResolver:
    """Evaluate PAC script files and resolve proxies."""

    @staticmethod
    def _parse_proxy_host(host_str):
        if False:
            while True:
                i = 10
        (host, _colon, port_str) = host_str.partition(':')
        try:
            port = int(port_str)
        except ValueError:
            raise ParseProxyError('Invalid port number')
        return (host, port)

    @staticmethod
    def _parse_proxy_entry(proxy_str):
        if False:
            return 10
        'Parse one proxy string entry, as described in PAC specification.'
        config = [c.strip() for c in proxy_str.split(' ') if c]
        if not config:
            raise ParseProxyError('Empty proxy entry')
        if config[0] == 'DIRECT':
            if len(config) != 1:
                raise ParseProxyError('Invalid number of parameters for ' + 'DIRECT')
            return QNetworkProxy(QNetworkProxy.ProxyType.NoProxy)
        elif config[0] == 'PROXY':
            if len(config) != 2:
                raise ParseProxyError('Invalid number of parameters for PROXY')
            (host, port) = PACResolver._parse_proxy_host(config[1])
            return QNetworkProxy(QNetworkProxy.ProxyType.HttpProxy, host, port)
        elif config[0] in ['SOCKS', 'SOCKS5']:
            if len(config) != 2:
                raise ParseProxyError('Invalid number of parameters for SOCKS')
            (host, port) = PACResolver._parse_proxy_host(config[1])
            return QNetworkProxy(QNetworkProxy.ProxyType.Socks5Proxy, host, port)
        else:
            err = 'Unknown proxy type: {}'
            raise ParseProxyError(err.format(config[0]))

    @staticmethod
    def _parse_proxy_string(proxy_str):
        if False:
            i = 10
            return i + 15
        proxies = proxy_str.split(';')
        return [PACResolver._parse_proxy_entry(x) for x in proxies]

    def _evaluate(self, js_code, js_file):
        if False:
            while True:
                i = 10
        ret = self._engine.evaluate(js_code, js_file)
        if ret.isError():
            err = 'JavaScript error while evaluating PAC file: {}'
            raise EvalProxyError(err.format(ret.toString()))

    def __init__(self, pac_str):
        if False:
            i = 10
            return i + 15
        'Create a PAC resolver.\n\n        Args:\n            pac_str: JavaScript code containing PAC resolver.\n        '
        self._engine = QJSEngine()
        self._engine.installExtensions(QJSEngine.Extension.ConsoleExtension)
        self._ctx = _PACContext(self._engine)
        self._engine.globalObject().setProperty('PAC', self._engine.newQObject(self._ctx))
        self._evaluate(_PACContext.JS_DEFINITIONS, 'pac_js_definitions')
        self._evaluate(resources.read_file('javascript/pac_utils.js'), 'pac_utils')
        proxy_config = self._engine.newObject()
        proxy_config.setProperty('bindings', self._engine.newObject())
        self._engine.globalObject().setProperty('ProxyConfig', proxy_config)
        self._evaluate(pac_str, 'pac')
        global_js_object = self._engine.globalObject()
        self._resolver = global_js_object.property('FindProxyForURL')
        if not self._resolver.isCallable():
            err = "Cannot resolve FindProxyForURL function, got '{}' instead"
            raise EvalProxyError(err.format(self._resolver.toString()))

    def resolve(self, query, from_file=False):
        if False:
            i = 10
            return i + 15
        'Resolve a proxy via PAC.\n\n        Args:\n            query: QNetworkProxyQuery.\n            from_file: Whether the proxy info is coming from a file.\n\n        Return:\n            A list of QNetworkProxy objects in order of preference.\n        '
        qtutils.ensure_valid(query.url())
        string_flags: urlutils.UrlFlagsType
        if from_file:
            string_flags = QUrl.ComponentFormattingOption.PrettyDecoded
        else:
            string_flags = QUrl.UrlFormattingOption.RemoveUserInfo
            if query.url().scheme() == 'https':
                https_opts = QUrl.UrlFormattingOption.RemovePath | QUrl.UrlFormattingOption.RemoveQuery
                if machinery.IS_QT5:
                    string_flags |= cast(QUrl.UrlFormattingOption, https_opts)
                else:
                    string_flags |= https_opts
        result = self._resolver.call([query.url().toString(string_flags), query.peerHostName()])
        result_str = result.toString()
        if not result.isString():
            err = "Got strange value from FindProxyForURL: '{}'"
            raise EvalProxyError(err.format(result_str))
        return self._parse_proxy_string(result_str)

class PACFetcher(QObject):
    """Asynchronous fetcher of PAC files."""
    finished = pyqtSignal()

    def __init__(self, url, parent=None):
        if False:
            print('Hello World!')
        'Resolve a PAC proxy from URL.\n\n        Args:\n            url: QUrl of a PAC proxy.\n        '
        super().__init__(parent)
        pac_prefix = 'pac+'
        assert url.scheme().startswith(pac_prefix)
        url.setScheme(url.scheme()[len(pac_prefix):])
        self._pac_url = url
        with qtlog.disable_qt_msghandler():
            self._manager: Optional[QNetworkAccessManager] = QNetworkAccessManager()
        self._manager.setProxy(QNetworkProxy(QNetworkProxy.ProxyType.NoProxy))
        self._pac = None
        self._error_message = None
        self._reply = None

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return self._pac_url == other._pac_url

    def __repr__(self):
        if False:
            while True:
                i = 10
        return utils.get_repr(self, url=self._pac_url, constructor=True)

    def fetch(self):
        if False:
            print('Hello World!')
        'Fetch the proxy from the remote URL.'
        assert self._manager is not None
        self._reply = self._manager.get(QNetworkRequest(self._pac_url))
        assert self._reply is not None
        self._reply.finished.connect(self._finish)

    @pyqtSlot()
    def _finish(self):
        if False:
            return 10
        assert self._reply is not None
        if self._reply.error() != QNetworkReply.NetworkError.NoError:
            error = "Can't fetch PAC file from URL, error code {}: {}"
            self._error_message = error.format(self._reply.error(), self._reply.errorString())
            log.network.error(self._error_message)
        else:
            try:
                pacscript = bytes(self._reply.readAll()).decode('utf-8')
            except UnicodeError as e:
                error = 'Invalid encoding of a PAC file: {}'
                self._error_message = error.format(e)
                log.network.exception(self._error_message)
                return
            try:
                self._pac = PACResolver(pacscript)
                log.network.debug('Successfully evaluated PAC file.')
            except EvalProxyError as e:
                error = 'Error in PAC evaluation: {}'
                self._error_message = error.format(e)
                log.network.exception(self._error_message)
        self._manager = None
        self._reply = None
        self.finished.emit()

    def _wait(self):
        if False:
            i = 10
            return i + 15
        'Wait until a reply from the remote server is received.'
        if self._manager is not None:
            loop = qtutils.EventLoop()
            self.finished.connect(loop.quit)
            loop.exec()

    def fetch_error(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if PAC script is successfully fetched.\n\n        Return None iff PAC script is downloaded and evaluated successfully,\n        error string otherwise.\n        '
        self._wait()
        return self._error_message

    def resolve(self, query):
        if False:
            for i in range(10):
                print('nop')
        'Resolve a query via PAC.\n\n        Args: QNetworkProxyQuery.\n\n        Return a list of QNetworkProxy objects in order of preference.\n        '
        self._wait()
        assert self._pac is not None
        from_file = self._pac_url.scheme() == 'file'
        try:
            return self._pac.resolve(query, from_file=from_file)
        except (EvalProxyError, ParseProxyError) as e:
            log.network.exception('Error in PAC resolution: {}.'.format(e))
            error_host = 'pac-resolve-error.qutebrowser.invalid'
            return [QNetworkProxy(QNetworkProxy.ProxyType.HttpProxy, error_host, 9)]
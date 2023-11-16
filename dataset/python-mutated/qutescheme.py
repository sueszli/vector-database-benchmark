"""Backend-independent qute://* code.

Module attributes:
    pyeval_output: The output of the last :pyeval command.
    _HANDLERS: The handlers registered via decorators.
"""
import sys
import html
import json
import os
import time
import textwrap
import urllib
import collections
import secrets
from typing import TypeVar, Callable, Dict, List, Optional, Union, Sequence, Tuple
from qutebrowser.qt.core import QUrlQuery, QUrl
import qutebrowser
from qutebrowser.browser import pdfjs, downloads, history
from qutebrowser.config import config, configdata, configexc
from qutebrowser.utils import version, utils, jinja, log, message, docutils, resources, objreg, standarddir
from qutebrowser.misc import guiprocess, quitter
from qutebrowser.qt import sip
pyeval_output = ':pyeval was never called'
csrf_token = None
_HANDLERS = {}

class Error(Exception):
    """Exception for generic errors on a qute:// page."""

class NotFoundError(Error):
    """Raised when the given URL was not found."""

class SchemeOSError(Error):
    """Raised when there was an OSError inside a handler."""

class UrlInvalidError(Error):
    """Raised when an invalid URL was opened."""

class RequestDeniedError(Error):
    """Raised when the request is forbidden."""

class Redirect(Exception):
    """Exception to signal a redirect should happen.

    Attributes:
        url: The URL to redirect to, as a QUrl.
    """

    def __init__(self, url: QUrl):
        if False:
            return 10
        super().__init__(url.toDisplayString())
        self.url = url
_HandlerRet = Tuple[str, Union[str, bytes]]
_HandlerCallable = Callable[[QUrl], _HandlerRet]
_Handler = TypeVar('_Handler', bound=_HandlerCallable)

class add_handler:
    """Decorator to register a qute://* URL handler.

    Attributes:
        _name: The 'foo' part of qute://foo
    """

    def __init__(self, name: str) -> None:
        if False:
            return 10
        self._name = name
        self._function: Optional[_HandlerCallable] = None

    def __call__(self, function: _Handler) -> _Handler:
        if False:
            for i in range(10):
                print('nop')
        self._function = function
        _HANDLERS[self._name] = self.wrapper
        return function

    def wrapper(self, url: QUrl) -> _HandlerRet:
        if False:
            print('Hello World!')
        'Call the underlying function.'
        assert self._function is not None
        return self._function(url)

def data_for_url(url: QUrl) -> Tuple[str, bytes]:
    if False:
        while True:
            i = 10
    'Get the data to show for the given URL.\n\n    Args:\n        url: The QUrl to show.\n\n    Return:\n        A (mimetype, data) tuple.\n    '
    norm_url = url.adjusted(QUrl.UrlFormattingOption.NormalizePathSegments | QUrl.UrlFormattingOption.StripTrailingSlash)
    if norm_url != url:
        raise Redirect(norm_url)
    path = url.path()
    host = url.host()
    query = url.query()
    log.misc.debug('url: {}, path: {}, host {}'.format(url.toDisplayString(), path, host))
    if not path or not host:
        new_url = QUrl()
        new_url.setScheme('qute')
        if host:
            new_url.setHost(host)
        else:
            new_url.setHost(path)
        new_url.setPath('/')
        if query:
            new_url.setQuery(query)
        if new_url.host():
            raise Redirect(new_url)
    try:
        handler = _HANDLERS[host]
    except KeyError:
        raise NotFoundError('No handler found for {}'.format(url.toDisplayString()))
    try:
        (mimetype, data) = handler(url)
    except OSError as e:
        raise SchemeOSError(e)
    assert mimetype is not None, url
    if mimetype == 'text/html' and isinstance(data, str):
        data = data.encode('utf-8', errors='xmlcharrefreplace')
    assert isinstance(data, bytes)
    return (mimetype, data)

@add_handler('bookmarks')
def qute_bookmarks(_url: QUrl) -> _HandlerRet:
    if False:
        print('Hello World!')
    'Handler for qute://bookmarks. Display all quickmarks / bookmarks.'
    bookmarks = sorted(objreg.get('bookmark-manager').marks.items(), key=lambda x: x[1])
    quickmarks = sorted(objreg.get('quickmark-manager').marks.items(), key=lambda x: x[0])
    src = jinja.render('bookmarks.html', title='Bookmarks', bookmarks=bookmarks, quickmarks=quickmarks)
    return ('text/html', src)

@add_handler('tabs')
def qute_tabs(_url: QUrl) -> _HandlerRet:
    if False:
        print('Hello World!')
    'Handler for qute://tabs. Display information about all open tabs.'
    tabs: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
    for (win_id, window) in objreg.window_registry.items():
        if sip.isdeleted(window):
            continue
        tabbed_browser = objreg.get('tabbed-browser', scope='window', window=win_id)
        for tab in tabbed_browser.widgets():
            if tab.url() not in [QUrl('qute://tabs/'), QUrl('qute://tabs')]:
                urlstr = tab.url().toDisplayString()
                tabs[str(win_id)].append((tab.title(), urlstr))
    src = jinja.render('tabs.html', title='Tabs', tab_list_by_window=tabs)
    return ('text/html', src)

def history_data(start_time: float, offset: int=None) -> Sequence[Dict[str, Union[str, int]]]:
    if False:
        return 10
    'Return history data.\n\n    Arguments:\n        start_time: select history starting from this timestamp.\n        offset: number of items to skip\n    '
    start_time = int(start_time)
    if offset is not None:
        entries = history.web_history.entries_before(start_time, limit=1000, offset=offset)
    else:
        end_time = start_time - 24 * 60 * 60
        entries = history.web_history.entries_between(end_time, start_time)
    return [{'url': e.url, 'title': html.escape(e.title) or html.escape(e.url), 'time': e.atime} for e in entries]

@add_handler('history')
def qute_history(url: QUrl) -> _HandlerRet:
    if False:
        while True:
            i = 10
    'Handler for qute://history. Display and serve history.'
    if url.path() == '/data':
        q_offset = QUrlQuery(url).queryItemValue('offset')
        try:
            offset = int(q_offset) if q_offset else None
        except ValueError:
            raise UrlInvalidError('Query parameter offset is invalid')
        q_start_time = QUrlQuery(url).queryItemValue('start_time')
        try:
            start_time = float(q_start_time) if q_start_time else time.time()
        except ValueError:
            raise UrlInvalidError('Query parameter start_time is invalid')
        return ('text/html', json.dumps(history_data(start_time, offset)))
    else:
        return ('text/html', jinja.render('history.html', title='History', gap_interval=config.val.history_gap_interval))

@add_handler('javascript')
def qute_javascript(url: QUrl) -> _HandlerRet:
    if False:
        return 10
    'Handler for qute://javascript.\n\n    Return content of file given as query parameter.\n    '
    path = url.path()
    if path:
        path = 'javascript' + os.sep.join(path.split('/'))
        return ('text/html', resources.read_file(path))
    else:
        raise UrlInvalidError('No file specified')

@add_handler('pyeval')
def qute_pyeval(_url: QUrl) -> _HandlerRet:
    if False:
        for i in range(10):
            print('nop')
    'Handler for qute://pyeval.'
    src = jinja.render('pre.html', title='pyeval', content=pyeval_output)
    return ('text/html', src)

@add_handler('process')
def qute_process(url: QUrl) -> _HandlerRet:
    if False:
        for i in range(10):
            print('nop')
    'Handler for qute://process.'
    path = url.path()[1:]
    try:
        pid = int(path)
    except ValueError:
        raise UrlInvalidError(f'Invalid PID {path}')
    try:
        proc = guiprocess.all_processes[pid]
    except KeyError:
        raise NotFoundError(f'No process {pid}')
    if proc is None:
        raise NotFoundError(f'Data for process {pid} got cleaned up.')
    src = jinja.render('process.html', title=f'Process {pid}', proc=proc)
    return ('text/html', src)

@add_handler('version')
@add_handler('verizon')
def qute_version(_url: QUrl) -> _HandlerRet:
    if False:
        return 10
    'Handler for qute://version.'
    src = jinja.render('version.html', title='Version info', version=version.version_info(), copyright=qutebrowser.__copyright__)
    return ('text/html', src)

@add_handler('log')
def qute_log(url: QUrl) -> _HandlerRet:
    if False:
        i = 10
        return i + 15
    "Handler for qute://log.\n\n    There are three query parameters:\n\n    - level: The minimum log level to print.\n    For example, qute://log?level=warning prints warnings and errors.\n    Level can be one of: vdebug, debug, info, warning, error, critical.\n\n    - plain: If given (and not 'false'), plaintext is shown.\n\n    - logfilter: A filter string like the --logfilter commandline argument\n      accepts.\n    "
    query = QUrlQuery(url)
    plain = query.hasQueryItem('plain') and query.queryItemValue('plain').lower() != 'false'
    if log.ram_handler is None:
        content = 'Log output was disabled.' if plain else None
    else:
        level = query.queryItemValue('level')
        if not level:
            level = 'vdebug'
        filter_str = query.queryItemValue('logfilter')
        try:
            logfilter = log.LogFilter.parse(filter_str, only_debug=False) if filter_str else None
        except log.InvalidLogFilterError as e:
            raise UrlInvalidError(e)
        content = log.ram_handler.dump_log(html=not plain, level=level, logfilter=logfilter)
    template = 'pre.html' if plain else 'log.html'
    src = jinja.render(template, title='log', content=content)
    return ('text/html', src)

@add_handler('gpl')
def qute_gpl(_url: QUrl) -> _HandlerRet:
    if False:
        i = 10
        return i + 15
    'Handler for qute://gpl. Return HTML content as string.'
    return ('text/html', resources.read_file('html/license.html'))

def _asciidoc_fallback_path(html_path: str) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    'Fall back to plaintext asciidoc if the HTML is unavailable.'
    path = html_path.replace('.html', '.asciidoc')
    try:
        return resources.read_file(path)
    except OSError:
        return None

@add_handler('help')
def qute_help(url: QUrl) -> _HandlerRet:
    if False:
        i = 10
        return i + 15
    'Handler for qute://help.'
    urlpath = url.path()
    if not urlpath or urlpath == '/':
        urlpath = 'index.html'
    else:
        urlpath = urlpath.lstrip('/')
    if not docutils.docs_up_to_date(urlpath):
        message.error('Your documentation is outdated! Please re-run scripts/asciidoc2html.py.')
    path = 'html/doc/{}'.format(urlpath)
    if not urlpath.endswith('.html'):
        try:
            bdata = resources.read_file_binary(path)
        except OSError as e:
            raise SchemeOSError(e)
        mimetype = utils.guess_mimetype(urlpath)
        return (mimetype, bdata)
    try:
        data = resources.read_file(path)
    except OSError:
        asciidoc = _asciidoc_fallback_path(path)
        if asciidoc is None:
            raise
        preamble = textwrap.dedent("\n            There was an error loading the documentation!\n\n            This most likely means the documentation was not generated\n            properly. If you are running qutebrowser from the git repository,\n            please (re)run scripts/asciidoc2html.py and reload this page.\n\n            If you're running a released version this is a bug, please use\n            :report to report it.\n\n            Falling back to the plaintext version.\n\n            ---------------------------------------------------------------\n\n\n        ")
        return ('text/plain', (preamble + asciidoc).encode('utf-8'))
    else:
        return ('text/html', data)

def _qute_settings_set(url: QUrl) -> _HandlerRet:
    if False:
        return 10
    'Handler for qute://settings/set.'
    query = QUrlQuery(url)
    option = query.queryItemValue('option', QUrl.ComponentFormattingOption.FullyDecoded)
    value = query.queryItemValue('value', QUrl.ComponentFormattingOption.FullyDecoded)
    if option == 'content.javascript.enabled' and value == 'false':
        msg = 'Refusing to disable javascript via qute://settings as it needs javascript support.'
        message.error(msg)
        return ('text/html', b'error: ' + msg.encode('utf-8'))
    try:
        config.instance.set_str(option, value, save_yaml=True)
        return ('text/html', b'ok')
    except configexc.Error as e:
        message.error(str(e))
        return ('text/html', b'error: ' + str(e).encode('utf-8'))

@add_handler('settings')
def qute_settings(url: QUrl) -> _HandlerRet:
    if False:
        while True:
            i = 10
    'Handler for qute://settings. View/change qute configuration.'
    global csrf_token
    if url.path() == '/set':
        if url.password() != csrf_token:
            message.error('Invalid CSRF token for qute://settings!')
            raise RequestDeniedError('Invalid CSRF token!')
        if quitter.instance.is_shutting_down:
            log.config.debug('Ignoring /set request during shutdown')
            return ('text/html', b'error: ignored')
        return _qute_settings_set(url)
    csrf_token = secrets.token_urlsafe()
    src = jinja.render('settings.html', title='settings', configdata=configdata, confget=config.instance.get_str, csrf_token=csrf_token)
    return ('text/html', src)

@add_handler('bindings')
def qute_bindings(_url: QUrl) -> _HandlerRet:
    if False:
        while True:
            i = 10
    'Handler for qute://bindings. View keybindings.'
    bindings = {}
    defaults = config.val.bindings.default
    config_modes = set(defaults.keys()).union(config.val.bindings.commands)
    config_modes.remove('normal')
    modes = ['normal'] + sorted(config_modes)
    for mode in modes:
        bindings[mode] = config.key_instance.get_bindings_for(mode)
    src = jinja.render('bindings.html', title='Bindings', bindings=bindings)
    return ('text/html', src)

@add_handler('back')
def qute_back(url: QUrl) -> _HandlerRet:
    if False:
        for i in range(10):
            print('nop')
    'Handler for qute://back.\n\n    Simple page to free ram / lazy load a site, goes back on focusing the tab.\n    '
    src = jinja.render('back.html', title='Suspended: ' + urllib.parse.unquote(url.fragment()))
    return ('text/html', src)

@add_handler('configdiff')
def qute_configdiff(url: QUrl) -> _HandlerRet:
    if False:
        i = 10
        return i + 15
    'Handler for qute://configdiff.'
    include_hidden = QUrlQuery(url).queryItemValue('include_hidden') == 'true'
    dump = config.instance.dump_userconfig(include_hidden=include_hidden)
    return ('text/plain', dump.encode('utf-8'))

@add_handler('pastebin-version')
def qute_pastebin_version(_url: QUrl) -> _HandlerRet:
    if False:
        for i in range(10):
            print('nop')
    'Handler that pastebins the version string.'
    version.pastebin_version()
    return ('text/plain', b'Paste called.')

def _pdf_path(filename: str) -> str:
    if False:
        while True:
            i = 10
    'Get the path of a temporary PDF file.'
    return os.path.join(downloads.temp_download_manager.get_tmpdir().name, filename)

@add_handler('pdfjs')
def qute_pdfjs(url: QUrl) -> _HandlerRet:
    if False:
        for i in range(10):
            print('nop')
    'Handler for qute://pdfjs.\n\n    Return the pdf.js viewer or redirect to original URL if the file does not\n    exist.\n    '
    if url.path() == '/file':
        filename = QUrlQuery(url).queryItemValue('filename')
        if not filename:
            raise UrlInvalidError('Missing filename')
        if '/' in filename or os.sep in filename:
            raise RequestDeniedError('Path separator in filename.')
        path = _pdf_path(filename)
        with open(path, 'rb') as f:
            data = f.read()
        mimetype = utils.guess_mimetype(filename, fallback=True)
        return (mimetype, data)
    if url.path() == '/web/viewer.html':
        query = QUrlQuery(url)
        filename = query.queryItemValue('filename')
        if not filename:
            raise UrlInvalidError('Missing filename')
        path = _pdf_path(filename)
        if not os.path.isfile(path):
            source = query.queryItemValue('source')
            if not source:
                raise UrlInvalidError('Missing source')
            raise Redirect(QUrl(source))
        data = pdfjs.generate_pdfjs_page(filename, url)
        return ('text/html', data)
    try:
        data = pdfjs.get_pdfjs_res(url.path())
    except pdfjs.PDFJSNotFound as e:
        log.misc.warning('pdfjs resource requested but not found: {}'.format(e.path))
        raise NotFoundError("Can't find pdfjs resource '{}'".format(e.path))
    mimetype = utils.guess_mimetype(url.fileName(), fallback=True)
    return (mimetype, data)

@add_handler('warning')
def qute_warning(url: QUrl) -> _HandlerRet:
    if False:
        while True:
            i = 10
    'Handler for qute://warning.'
    path = url.path()
    if path == '/webkit':
        src = jinja.render('warning-webkit.html', title='QtWebKit backend warning')
    elif path == '/sessions':
        src = jinja.render('warning-sessions.html', title='Qt 5.15 sessions warning', datadir=standarddir.data(), sep=os.sep)
    elif path == '/qt5':
        is_venv = hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix
        src = jinja.render('warning-qt5.html', title='Switch to Qt 6', is_venv=is_venv, prefix=sys.prefix)
    else:
        raise NotFoundError('Invalid warning page {}'.format(path))
    return ('text/html', src)

@add_handler('resource')
def qute_resource(url: QUrl) -> _HandlerRet:
    if False:
        print('Hello World!')
    'Handler for qute://resource.'
    path = url.path().lstrip('/')
    mimetype = utils.guess_mimetype(path, fallback=True)
    try:
        data = resources.read_file_binary(path)
    except FileNotFoundError as e:
        raise NotFoundError(str(e))
    return (mimetype, data)

@add_handler('start')
def qute_start(_url: QUrl) -> _HandlerRet:
    if False:
        while True:
            i = 10
    'Handler for qute://start.'
    bookmarks = sorted(objreg.get('bookmark-manager').marks.items(), key=lambda x: x[1])
    quickmarks = sorted(objreg.get('quickmark-manager').marks.items(), key=lambda x: x[0])
    searchurl = config.val.url.searchengines['DEFAULT']
    page = jinja.render('startpage.html', title='Welcome to qutebrowser', bookmarks=bookmarks, search_url=searchurl, quickmarks=quickmarks)
    return ('text/html', page)
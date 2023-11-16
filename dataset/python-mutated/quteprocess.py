"""Fixtures to run qutebrowser in a QProcess and communicate."""
import pathlib
import re
import sys
import time
import datetime
import logging
import tempfile
import contextlib
import itertools
import collections
import json
import yaml
import pytest
from qutebrowser.qt.core import pyqtSignal, QUrl, QPoint
from qutebrowser.qt.gui import QImage, QColor
from qutebrowser.misc import ipc
from qutebrowser.utils import log, utils, javascript
from helpers import testutils
from end2end.fixtures import testprocess
instance_counter = itertools.count()

def is_ignored_qt_message(pytestconfig, message):
    if False:
        print('Hello World!')
    'Check if the message is listed in qt_log_ignore.'
    regexes = pytestconfig.getini('qt_log_ignore')
    return any((re.search(regex, message) for regex in regexes))

def is_ignored_lowlevel_message(message):
    if False:
        i = 10
        return i + 15
    'Check if we want to ignore a lowlevel process output.'
    ignored_messages = ['Fontconfig error: Cannot load default config file: No such file: (null)', 'Fontconfig error: Cannot load default config file', '----- Certificate i=0 (*,CN=localhost,O=qutebrowser test certificate) -----', 'ERROR: No matching issuer found', '", source: userscript:_qute_stylesheet (*)', 'QPaintDevice: Cannot destroy paint device that is being painted', 'libva error: vaGetDriverNameByIndex() failed with unknown libva error, driver_name = (null)', 'libva error: vaGetDriverNames() failed with unknown libva error']
    return any((testutils.pattern_match(pattern=pattern, value=message) for pattern in ignored_messages))

def is_ignored_chromium_message(line):
    if False:
        print('Hello World!')
    msg_re = re.compile('\n        \\[\n        (\\d+:\\d+:)?  # Process/Thread ID\n        \\d{4}/[\\d.]+:  # MMDD/Time\n        (?P<loglevel>[A-Z]+):  # Log level\n        [^ :]+    # filename / line\n        \\]\n        \\ (?P<message>.*)  # message\n    ', re.VERBOSE)
    match = msg_re.fullmatch(line)
    if match is None:
        return False
    if match.group('loglevel') == 'INFO':
        return True
    message = match.group('message')
    ignored_messages = ['SharedImageManager::ProduceGLTexture: Trying to produce a representation from a non-existent mailbox. *', '[.DisplayCompositor]GL ERROR :GL_INVALID_OPERATION : DoCreateAndTexStorage2DSharedImageINTERNAL: invalid mailbox name', '[.DisplayCompositor]GL ERROR :GL_INVALID_OPERATION : DoBeginSharedImageAccessCHROMIUM: bound texture is not a shared image', '[.DisplayCompositor]RENDER WARNING: texture bound to texture unit 0 is not renderable. It might be non-power-of-2 or have incompatible texture filtering (maybe)?', '[.DisplayCompositor]GL ERROR :GL_INVALID_OPERATION : DoEndSharedImageAccessCHROMIUM: bound texture is not a shared image', 'Could not bind NETLINK socket: Address already in use (98)', 'mDNS responder manager failed to start.', 'The mDNS responder manager is not started yet.', 'handshake failed; returned -1, SSL error code 1, net_error -202', 'ContextResult::kTransientFailure: Failed to send *CreateCommandBuffer.', 'GPU state invalid after WaitForGetOffsetInRange.', 'Unable to map Index file', 'CertVerifyProcBuiltin for localhost failed:', 'Message 4 rejected by interface blink.mojom.WidgetHost', 'GpuChannelHost failed to create command buffer.', 'Failed to create temporary file to update *user_prefs.json: No such file or directory (2)', 'Failed to create temporary file to update *Network Persistent State: No such file or directory (2)', 'Could not open platform files for entry.', 'Dropping message on closed channel.', 'RED codec red is missing an associated payload type.', 'SetError: {code=4, message="MEDIA_ELEMENT_ERROR: Media load rejected by URL safety check"}', 'Old/orphaned temporary reference to SurfaceId(FrameSinkId[](*, *), LocalSurfaceId(*, *, *...))', 'Input request on unbound interface', 'ReadData failed: 0', 'Message * rejected by interface blink.mojom.Widget*', 'dri3 extension not supported.']
    return any((testutils.pattern_match(pattern=pattern, value=message) for pattern in ignored_messages))

class LogLine(testprocess.Line):
    """A parsed line from the qutebrowser log output.

    Attributes:
        timestamp/loglevel/category/module/function/line/message/levelname:
            Parsed from the log output.
        expected: Whether the message was expected or not.
    """

    def __init__(self, pytestconfig, data):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(data)
        try:
            line = json.loads(data)
        except ValueError:
            raise testprocess.InvalidLine(data)
        if not isinstance(line, dict):
            raise testprocess.InvalidLine(data)
        self.timestamp = datetime.datetime.fromtimestamp(line['created'])
        self.msecs = line['msecs']
        self.loglevel = line['levelno']
        self.levelname = line['levelname']
        self.category = line['name']
        self.module = line['module']
        self.function = line['funcName']
        self.line = line['lineno']
        if self.function is None and self.line == 0:
            self.line = None
        self.traceback = line.get('traceback')
        self.message = line['message']
        self.expected = is_ignored_qt_message(pytestconfig, self.message)
        self.use_color = False

    def __str__(self):
        if False:
            print('Hello World!')
        return self.formatted_str(colorized=self.use_color)

    def formatted_str(self, colorized=True):
        if False:
            print('Hello World!')
        'Return a formatted colorized line.\n\n        This returns a line like qute without --json-logging would produce.\n\n        Args:\n            colorized: If True, ANSI color codes will be embedded.\n        '
        r = logging.LogRecord(self.category, self.loglevel, '', self.line, self.message, (), None)
        if self.line is None:
            r.line = 0
        r.created = self.timestamp.timestamp()
        r.msecs = self.msecs
        r.module = self.module
        r.funcName = self.function
        format_str = log.EXTENDED_FMT
        format_str = format_str.replace('{asctime:8}', '{asctime:8}.{msecs:03.0f}')
        if self.expected and self.loglevel > logging.INFO:
            new_color = '{' + log.LOG_COLORS['DEBUG'] + '}'
            format_str = format_str.replace('{log_color}', new_color)
            format_str = re.sub('{levelname:(\\d*)}', '{levelname} (expected)', format_str)
        formatter = log.ColoredFormatter(format_str, log.DATEFMT, '{', use_colors=colorized)
        result = formatter.format(r)
        if self.traceback is not None:
            result += '\n' + self.traceback
        return result

class QuteProc(testprocess.Process):
    """A running qutebrowser process used for tests.

    Attributes:
        _ipc_socket: The IPC socket of the started instance.
        _webengine: Whether to use QtWebEngine
        basedir: The base directory for this instance.
        request: The request object for the current test.
        _instance_id: A unique ID for this QuteProc instance
        _run_counter: A counter to get a unique ID for each run.

    Signals:
        got_error: Emitted when there was an error log line.
    """
    got_error = pyqtSignal()
    KEYS = ['timestamp', 'loglevel', 'category', 'module', 'function', 'line', 'message']

    def __init__(self, request, *, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(request, parent)
        self._ipc_socket = None
        self.basedir = None
        self._instance_id = next(instance_counter)
        self._run_counter = itertools.count()
        self._screenshot_counters = collections.defaultdict(itertools.count)

    def _process_line(self, log_line):
        if False:
            i = 10
            return i + 15
        "Check if the line matches any initial lines we're interested in."
        start_okay_message = "load status for <qutebrowser.browser.* tab_id=0 url='about:blank'>: LoadStatus.success"
        if log_line.category == 'ipc' and log_line.message.startswith('Listening as '):
            self._ipc_socket = log_line.message.split(' ', maxsplit=2)[2]
        elif log_line.category == 'webview' and testutils.pattern_match(pattern=start_okay_message, value=log_line.message):
            log_line.waited_for = True
            self.ready.emit()
        elif log_line.category == 'init' and log_line.module == 'standarddir' and (log_line.function == 'init') and log_line.message.startswith('Base directory:'):
            self.basedir = log_line.message.split(':', maxsplit=1)[1].strip()
        elif self._is_error_logline(log_line):
            self.got_error.emit()

    def _parse_line(self, line):
        if False:
            while True:
                i = 10
        try:
            log_line = LogLine(self.request.config, line)
        except testprocess.InvalidLine:
            if not line.strip():
                return None
            elif is_ignored_qt_message(self.request.config, line) or is_ignored_lowlevel_message(line) or is_ignored_chromium_message(line) or list(self.request.node.iter_markers('no_invalid_lines')):
                self._log('IGNORED: {}'.format(line))
                return None
            else:
                raise
        log_line.use_color = self.request.config.getoption('--color') != 'no'
        verbose = self.request.config.getoption('--verbose')
        if log_line.loglevel > logging.VDEBUG or verbose:
            self._log(log_line)
        self._process_line(log_line)
        return log_line

    def _executable_args(self):
        if False:
            while True:
                i = 10
        profile = self.request.config.getoption('--qute-profile-subprocs')
        if hasattr(sys, 'frozen'):
            if profile:
                raise RuntimeError("Can't profile with sys.frozen!")
            executable = str(pathlib.Path(sys.executable).parent / 'qutebrowser')
            args = []
        else:
            executable = sys.executable
            if profile:
                profile_dir = pathlib.Path.cwd() / 'prof'
                profile_id = '{}_{}'.format(self._instance_id, next(self._run_counter))
                profile_file = profile_dir / '{}.pstats'.format(profile_id)
                profile_dir.mkdir(exist_ok=True)
                args = [str(pathlib.Path('scripts') / 'dev' / 'run_profile.py'), '--profile-tool', 'none', '--profile-file', str(profile_file)]
            else:
                args = ['-bb', '-m', 'qutebrowser']
        return (executable, args)

    def _default_args(self):
        if False:
            print('Hello World!')
        backend = 'webengine' if self.request.config.webengine else 'webkit'
        args = ['--debug', '--no-err-windows', '--temp-basedir', '--json-logging', '--loglevel', 'vdebug', '--backend', backend, '--debug-flag', 'no-sql-history', '--debug-flag', 'werror', '--debug-flag', 'test-notification-service']
        if self.request.config.webengine and testutils.disable_seccomp_bpf_sandbox():
            args += testutils.DISABLE_SECCOMP_BPF_ARGS
        args.append('about:blank')
        return args

    def path_to_url(self, path, *, port=None, https=False):
        if False:
            for i in range(10):
                print('nop')
        'Get a URL based on a filename for the localhost webserver.\n\n        URLs like about:... and qute:... are handled specially and returned\n        verbatim.\n        '
        special_schemes = ['about:', 'qute:', 'chrome:', 'view-source:', 'data:', 'http:', 'https:', 'file:']
        server = self.request.getfixturevalue('server')
        server_port = server.port if port is None else port
        if any((path.startswith(scheme) for scheme in special_schemes)):
            path = path.replace('(port)', str(server_port))
            return path
        else:
            return '{}://localhost:{}/{}'.format('https' if https else 'http', server_port, path if path != '/' else '')

    def wait_for_js(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Wait for the given javascript console message.\n\n        Return:\n            The LogLine.\n        '
        line = self.wait_for(category='js', message='[*] {}'.format(message))
        line.expected = True
        return line

    def wait_scroll_pos_changed(self, x=None, y=None):
        if False:
            return 10
        'Wait until a "Scroll position changed" message was found.\n\n        With QtWebEngine, on older Qt versions which lack\n        QWebEnginePage.scrollPositionChanged, this also skips the test.\n        '
        __tracebackhide__ = lambda e: e.errisinstance(testprocess.WaitForTimeout)
        if x is None and y is not None or (y is None and x is not None):
            raise ValueError('Either both x/y or neither must be given!')
        if x is None and y is None:
            point = 'Py*.QtCore.QPoint(*, *)'
        elif x == '0' and y == '0':
            point = 'Py*.QtCore.QPoint()'
        else:
            point = 'Py*.QtCore.QPoint({}, {})'.format(x, y)
        self.wait_for(category='webview', message='Scroll position changed to ' + point)

    def wait_for(self, timeout=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Extend wait_for to add divisor if a test is xfailing.'
        __tracebackhide__ = lambda e: e.errisinstance(testprocess.WaitForTimeout)
        xfail = self.request.node.get_closest_marker('xfail')
        if xfail and (not xfail.args or xfail.args[0]):
            kwargs['divisor'] = 10
        else:
            kwargs['divisor'] = 1
        return super().wait_for(timeout=timeout, **kwargs)

    def _is_error_logline(self, msg):
        if False:
            while True:
                i = 10
        'Check if the given LogLine is some kind of error message.'
        is_js_error = msg.category == 'js' and testutils.pattern_match(pattern='[*] [FAIL] *', value=msg.message)
        is_ddg_load = testutils.pattern_match(pattern="load status for <* tab_id=* url='*duckduckgo*'>: *", value=msg.message)
        is_log_error = msg.loglevel > logging.INFO and (not msg.message.startswith('Ignoring world ID')) and (not msg.message.startswith('Could not initialize QtNetwork SSL support.'))
        return is_log_error or is_js_error or is_ddg_load

    def _maybe_skip(self):
        if False:
            print('Hello World!')
        'Skip the test if [SKIP] lines were logged.'
        skip_texts = []
        for msg in self._data:
            if msg.category == 'js' and testutils.pattern_match(pattern='[*] [SKIP] *', value=msg.message):
                skip_texts.append(msg.message.partition(' [SKIP] ')[2])
        if skip_texts:
            pytest.skip(', '.join(skip_texts))

    def before_test(self):
        if False:
            for i in range(10):
                print('nop')
        'Clear settings before every test.'
        super().before_test()
        self.send_cmd(':config-clear')
        self._init_settings()
        self.clear_data()

    def _init_settings(self):
        if False:
            return 10
        'Adjust some qutebrowser settings after starting.'
        settings = [('messages.timeout', '0'), ('auto_save.interval', '0'), ('new_instance_open_target_window', 'last-opened')]
        for (opt, value) in settings:
            self.set_setting(opt, value)

    def after_test(self):
        if False:
            print('Hello World!')
        'Handle unexpected/skip logging and clean up after each test.'
        __tracebackhide__ = lambda e: e.errisinstance(pytest.fail.Exception)
        bad_msgs = [msg for msg in self._data if self._is_error_logline(msg) and (not msg.expected)]
        try:
            call = self.request.node.rep_call
        except AttributeError:
            pass
        else:
            if call.failed or hasattr(call, 'wasxfail') or call.skipped:
                super().after_test()
                return
        try:
            if bad_msgs:
                text = 'Logged unexpected errors:\n\n' + '\n'.join((str(e) for e in bad_msgs))
                pytest.fail(text, pytrace=False)
            else:
                self._maybe_skip()
        finally:
            super().after_test()

    def _wait_for_ipc(self):
        if False:
            return 10
        'Wait for an IPC message to arrive.'
        self.wait_for(category='ipc', module='ipc', function='on_ready_read', message='Read from socket *')

    @contextlib.contextmanager
    def disable_capturing(self):
        if False:
            for i in range(10):
                print('nop')
        capmanager = self.request.config.pluginmanager.getplugin('capturemanager')
        with capmanager.global_and_fixture_disabled():
            yield

    def _after_start(self):
        if False:
            print('Hello World!')
        'Wait before continuing if requested, e.g. for debugger attachment.'
        delay = self.request.config.getoption('--qute-delay-start')
        if delay:
            with self.disable_capturing():
                print(f'- waiting {delay}ms for quteprocess (PID: {self.proc.processId()})...')
            time.sleep(delay / 1000)

    def send_ipc(self, commands, target_arg=''):
        if False:
            print('Hello World!')
        'Send a raw command to the running IPC socket.'
        delay = self.request.config.getoption('--qute-delay')
        time.sleep(delay / 1000)
        assert self._ipc_socket is not None
        ipc.send_to_running_instance(self._ipc_socket, commands, target_arg)
        try:
            self._wait_for_ipc()
        except testprocess.WaitForTimeout:
            ipc.send_to_running_instance(self._ipc_socket, commands, target_arg)
            self._wait_for_ipc()

    def start(self, *args, **kwargs):
        if False:
            return 10
        try:
            super().start(*args, **kwargs)
        except testprocess.ProcessExited:
            is_dl_inconsistency = str(self.captured_log[-1]).endswith("_dl_allocate_tls_init: Assertion `listp->slotinfo[cnt].gen <= GL(dl_tls_generation)' failed!")
            if testutils.ON_CI and is_dl_inconsistency:
                self.captured_log = []
                self._log('NOTE: Restarted after libc DL inconsistency!')
                self.clear_data()
                super().start(*args, **kwargs)
            else:
                raise

    def send_cmd(self, command, count=None, invalid=False, *, escape=True):
        if False:
            i = 10
            return i + 15
        'Send a command to the running qutebrowser instance.\n\n        Args:\n            count: The count to pass to the command.\n            invalid: If True, we don\'t wait for "command called: ..." in the\n                     log and return None.\n            escape: Escape backslashes in the command\n\n        Return:\n            The parsed log line with "command called: ..." or None.\n        '
        __tracebackhide__ = lambda e: e.errisinstance(testprocess.WaitForTimeout)
        summary = command
        if count is not None:
            summary += ' (count {})'.format(count)
        self.log_summary(summary)
        if escape:
            command = command.replace('\\', '\\\\')
        if count is not None:
            command = ':cmd-run-with-count {} {}'.format(count, command.lstrip(':'))
        self.send_ipc([command])
        if invalid:
            return None
        else:
            return self.wait_for(category='commands', module='command', function='run', message='command called: *')

    def get_setting(self, opt, pattern=None):
        if False:
            for i in range(10):
                print('nop')
        'Get the value of a qutebrowser setting.'
        if pattern is None:
            cmd = ':set {}?'.format(opt)
        else:
            cmd = ':set -u {} {}?'.format(pattern, opt)
        self.send_cmd(cmd)
        msg = self.wait_for(loglevel=logging.INFO, category='message', message='{} = *'.format(opt))
        if pattern is None:
            return msg.message.split(' = ')[1]
        else:
            return msg.message.split(' = ')[1].split(' for ')[0]

    def set_setting(self, option, value):
        if False:
            for i in range(10):
                print('nop')
        value = value.replace('\\', '\\\\')
        value = value.replace('"', '\\"')
        self.send_cmd(':set -t "{}" "{}"'.format(option, value), escape=False)
        self.wait_for(category='config', message='Config option changed: *')

    @contextlib.contextmanager
    def temp_setting(self, opt, value):
        if False:
            return 10
        'Context manager to set a setting and reset it on exit.'
        old_value = self.get_setting(opt)
        self.set_setting(opt, value)
        yield
        self.set_setting(opt, old_value)

    def open_path(self, path, *, new_tab=False, new_bg_tab=False, new_window=False, private=False, as_url=False, port=None, https=False, wait=True):
        if False:
            i = 10
            return i + 15
        'Open the given path on the local webserver in qutebrowser.'
        url = self.path_to_url(path, port=port, https=https)
        self.open_url(url, new_tab=new_tab, new_bg_tab=new_bg_tab, new_window=new_window, private=private, as_url=as_url, wait=wait)

    def open_url(self, url, *, new_tab=False, new_bg_tab=False, new_window=False, private=False, as_url=False, wait=True):
        if False:
            print('Hello World!')
        'Open the given url in qutebrowser.'
        if sum((1 for opt in [new_tab, new_bg_tab, new_window, private, as_url] if opt)) > 1:
            raise ValueError('Conflicting options given!')
        if as_url:
            self.send_cmd(url, invalid=True)
            line = None
        elif new_tab:
            line = self.send_cmd(':open -t ' + url)
        elif new_bg_tab:
            line = self.send_cmd(':open -b ' + url)
        elif new_window:
            line = self.send_cmd(':open -w ' + url)
        elif private:
            line = self.send_cmd(':open -p ' + url)
        else:
            line = self.send_cmd(':open ' + url)
        if wait:
            self.wait_for_load_finished_url(url, after=line)

    def mark_expected(self, category=None, loglevel=None, message=None):
        if False:
            for i in range(10):
                print('nop')
        'Mark a given logging message as expected.'
        line = self.wait_for(category=category, loglevel=loglevel, message=message)
        line.expected = True

    def wait_for_load_finished_url(self, url, *, timeout=None, load_status='success', after=None):
        if False:
            print('Hello World!')
        'Wait until a URL has finished loading.'
        __tracebackhide__ = lambda e: e.errisinstance(testprocess.WaitForTimeout)
        if timeout is None:
            if testutils.ON_CI:
                timeout = 15000
            else:
                timeout = 5000
        qurl = QUrl(url)
        if not qurl.isValid():
            raise ValueError('Invalid URL {}: {}'.format(url, qurl.errorString()))
        url = utils.elide(qurl.toDisplayString(QUrl.ComponentFormattingOption.EncodeUnicode), 100)
        assert url
        pattern = re.compile("(load status for <qutebrowser\\.browser\\..* tab_id=\\d+ url='{url}/?'>: LoadStatus\\.{load_status}|fetch: Py.*\\.QtCore\\.QUrl\\('{url}'\\) -> .*)".format(load_status=re.escape(load_status), url=re.escape(url)))
        try:
            self.wait_for(message=pattern, timeout=timeout, after=after)
        except testprocess.WaitForTimeout:
            raise testprocess.WaitForTimeout('Timed out while waiting for {} to be loaded'.format(url))

    def wait_for_load_finished(self, path, *, port=None, https=False, timeout=None, load_status='success'):
        if False:
            for i in range(10):
                print('nop')
        'Wait until a path has finished loading.'
        __tracebackhide__ = lambda e: e.errisinstance(testprocess.WaitForTimeout)
        url = self.path_to_url(path, port=port, https=https)
        self.wait_for_load_finished_url(url, timeout=timeout, load_status=load_status)

    def get_session(self, flags='--with-private'):
        if False:
            return 10
        'Save the session and get the parsed session data.'
        with tempfile.TemporaryDirectory() as tdir:
            session = pathlib.Path(tdir) / 'session.yml'
            self.send_cmd(f':session-save {flags} "{session}"')
            self.wait_for(category='message', loglevel=logging.INFO, message=f'Saved session {session}.')
            data = session.read_text(encoding='utf-8')
        self._log('\nCurrent session data:\n' + data)
        return utils.yaml_load(data)

    def get_content(self, plain=True):
        if False:
            for i in range(10):
                print('nop')
        'Get the contents of the current page.'
        with tempfile.TemporaryDirectory() as tdir:
            path = pathlib.Path(tdir) / 'page'
            if plain:
                self.send_cmd(':debug-dump-page --plain "{}"'.format(path))
            else:
                self.send_cmd(':debug-dump-page "{}"'.format(path))
            self.wait_for(category='message', loglevel=logging.INFO, message='Dumped page to {}.'.format(path))
            return path.read_text(encoding='utf-8')

    def get_screenshot(self, *, probe_pos: QPoint=None, probe_color: QColor=testutils.Color(0, 0, 0)) -> QImage:
        if False:
            for i in range(10):
                print('nop')
        "Get a screenshot of the current page.\n\n        Arguments:\n            probe_pos: If given, only continue if the pixel at the given\n                       position isn't black (or whatever is specified by probe_color).\n        "
        for _ in range(5):
            tmp_path = self.request.getfixturevalue('tmp_path')
            counter = self._screenshot_counters[self.request.node.nodeid]
            path = tmp_path / f'screenshot-{next(counter)}.png'
            self.send_cmd(f':screenshot {path}')
            screenshot_msg = f'Screenshot saved to {path}'
            self.wait_for(message=screenshot_msg)
            print(screenshot_msg)
            img = QImage(str(path))
            assert not img.isNull()
            if probe_pos is None:
                return img
            probed_color = testutils.Color(img.pixelColor(probe_pos))
            if probed_color == probe_color:
                return img
            time.sleep(0.5)
        assert probed_color == probe_color, 'Color probing failed, values on last try:'
        raise utils.Unreachable()

    def press_keys(self, keys):
        if False:
            return 10
        'Press the given keys using :fake-key.'
        self.send_cmd(':fake-key -g "{}"'.format(keys))

    def click_element_by_text(self, text):
        if False:
            i = 10
            return i + 15
        'Click the element with the given text.'
        script = 'var _es = document.evaluate(\'//*[text()={text}]\', document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);if (_es.snapshotLength == 0) {{ console.log("qute:no elems"); }} else if (_es.snapshotLength > 1) {{ console.log("qute:ambiguous elems") }} else {{ console.log("qute:okay"); _es.snapshotItem(0).click() }}'.format(text=javascript.string_escape(_xpath_escape(text)))
        self.send_cmd(':jseval ' + script, escape=False)
        message = self.wait_for_js('qute:*').message
        if message.endswith('qute:no elems'):
            raise ValueError('No element with {!r} found'.format(text))
        if message.endswith('qute:ambiguous elems'):
            raise ValueError('Element with {!r} is not unique'.format(text))
        if not message.endswith('qute:okay'):
            raise ValueError('Invalid response from qutebrowser: {}'.format(message))

    def compare_session(self, expected, *, flags='--with-private'):
        if False:
            return 10
        'Compare the current sessions against the given template.\n\n        partial_compare is used, which means only the keys/values listed will\n        be compared.\n        '
        __tracebackhide__ = lambda e: e.errisinstance(pytest.fail.Exception)
        data = self.get_session(flags=flags)
        expected = yaml.load(expected, Loader=YamlLoader)
        outcome = testutils.partial_compare(data, expected)
        if not outcome:
            msg = 'Session comparison failed: {}'.format(outcome.error)
            msg += '\nsee stdout for details'
            pytest.fail(msg)

    def turn_on_scroll_logging(self, no_scroll_filtering=False):
        if False:
            while True:
                i = 10
        'Make sure all scrolling changes are logged.'
        cmd = ":debug-pyeval -q objects.debug_flags.add('{}')"
        if no_scroll_filtering:
            self.send_cmd(cmd.format('no-scroll-filtering'))
        self.send_cmd(cmd.format('log-scroll-pos'))

class YamlLoader(yaml.SafeLoader):
    """Custom YAML loader used in compare_session."""
YamlLoader.add_constructor('!ellipsis', lambda loader, node: ...)
YamlLoader.add_implicit_resolver('!ellipsis', re.compile('\\.\\.\\.'), None)

def _xpath_escape(text):
    if False:
        print('Hello World!')
    'Escape a string to be used in an XPath expression.\n\n    The resulting string should still be escaped with javascript.string_escape,\n    to prevent javascript from interpreting the quotes.\n\n    This function is needed because XPath does not provide any character\n    escaping mechanisms, so to get the string\n        "I\'m back", he said\n    you have to use concat like\n        concat(\'"I\', "\'m back", \'", he said\')\n\n    Args:\n        text: Text to escape\n\n    Return:\n        The string "escaped" as a concat() call.\n    '
    if "'" not in text or '"' not in text:
        return repr(text)
    parts = re.split('([\'"])', text)
    parts = [repr(part) for part in parts if part]
    return 'concat({})'.format(', '.join(parts))

@pytest.fixture(scope='module')
def quteproc_process(qapp, server, request):
    if False:
        i = 10
        return i + 15
    'Fixture for qutebrowser process which is started once per file.'
    proc = QuteProc(request)
    proc.start()
    yield proc
    proc.terminate()

@pytest.fixture
def quteproc(quteproc_process, server, request):
    if False:
        return 10
    'Per-test qutebrowser fixture which uses the per-file process.'
    request.node._quteproc_log = quteproc_process.captured_log
    quteproc_process.before_test()
    quteproc_process.request = request
    yield quteproc_process
    quteproc_process.after_test()

@pytest.fixture
def quteproc_new(qapp, server, request):
    if False:
        i = 10
        return i + 15
    'Per-test qutebrowser process to test invocations.'
    proc = QuteProc(request)
    request.node._quteproc_log = proc.captured_log
    yield proc
    proc.after_test()
    proc.terminate()
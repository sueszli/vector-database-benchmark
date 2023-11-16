import asyncio
import json
import locale
import traceback
import sys
import queue
from typing import NamedTuple, Optional
from .version import ELECTRUM_VERSION
from . import constants
from .i18n import _
from .util import make_aiohttp_session, error_text_str_to_safe_str
from .logging import describe_os_version, Logger, get_git_version

class CrashReportResponse(NamedTuple):
    status: Optional[str]
    text: str
    url: Optional[str]

class BaseCrashReporter(Logger):
    report_server = 'https://crashhub.electrum.org'
    issue_template = '<h2>Traceback</h2>\n<pre>\n{traceback}\n</pre>\n\n<h2>Additional information</h2>\n<ul>\n  <li>Electrum version: {app_version}</li>\n  <li>Python version: {python_version}</li>\n  <li>Operating system: {os}</li>\n  <li>Wallet type: {wallet_type}</li>\n  <li>Locale: {locale}</li>\n</ul>\n    '
    CRASH_MESSAGE = _('Something went wrong while executing Electrum.')
    CRASH_TITLE = _('Sorry!')
    REQUEST_HELP_MESSAGE = _('To help us diagnose and fix the problem, you can send us a bug report that contains useful debug information:')
    DESCRIBE_ERROR_MESSAGE = _('Please briefly describe what led to the error (optional):')
    ASK_CONFIRM_SEND = _('Do you want to send this report?')
    USER_COMMENT_PLACEHOLDER = _('Do not enter sensitive/private information here. The report will be visible on the public issue tracker.')

    def __init__(self, exctype, value, tb):
        if False:
            i = 10
            return i + 15
        Logger.__init__(self)
        self.exc_args = (exctype, value, tb)

    def send_report(self, asyncio_loop, proxy, *, timeout=None) -> CrashReportResponse:
        if False:
            while True:
                i = 10
        if constants.net.GENESIS[-4:] not in ['4943', 'e26f'] and '.electrum.org' in BaseCrashReporter.report_server:
            raise Exception(_('Missing report URL.'))
        report = self.get_traceback_info()
        report.update(self.get_additional_info())
        report = json.dumps(report)
        coro = self.do_post(proxy, BaseCrashReporter.report_server + '/crash.json', data=report)
        response = asyncio.run_coroutine_threadsafe(coro, asyncio_loop).result(timeout)
        self.logger.info(f'Crash report sent. Got response [DO NOT TRUST THIS MESSAGE]: {error_text_str_to_safe_str(response)}')
        response = json.loads(response)
        assert isinstance(response, dict), type(response)
        if (location := response.get('location')):
            assert isinstance(location, str)
            base_issues_url = constants.GIT_REPO_ISSUES_URL
            if not base_issues_url.endswith('/'):
                base_issues_url = base_issues_url + '/'
            if not location.startswith(base_issues_url):
                location = None
        ret = CrashReportResponse(status=response.get('status'), url=location, text=_('Thanks for reporting this issue!'))
        return ret

    async def do_post(self, proxy, url, data) -> str:
        async with make_aiohttp_session(proxy) as session:
            async with session.post(url, data=data, raise_for_status=True) as resp:
                return await resp.text()

    def get_traceback_info(self):
        if False:
            while True:
                i = 10
        exc_string = str(self.exc_args[1])
        stack = traceback.extract_tb(self.exc_args[2])
        readable_trace = self.__get_traceback_str_to_send()
        id = {'file': stack[-1].filename if len(stack) else '<no stack>', 'name': stack[-1].name if len(stack) else '<no stack>', 'type': self.exc_args[0].__name__}
        return {'exc_string': exc_string, 'stack': readable_trace, 'id': id}

    def get_additional_info(self):
        if False:
            while True:
                i = 10
        args = {'app_version': get_git_version() or ELECTRUM_VERSION, 'python_version': sys.version, 'os': describe_os_version(), 'wallet_type': 'unknown', 'locale': locale.getdefaultlocale()[0] or '?', 'description': self.get_user_description()}
        try:
            args['wallet_type'] = self.get_wallet_type()
        except Exception:
            pass
        return args

    def __get_traceback_str_to_send(self) -> str:
        if False:
            print('Hello World!')
        return ''.join(traceback.format_exception(*self.exc_args))

    def _get_traceback_str_to_display(self) -> str:
        if False:
            return 10
        return self.__get_traceback_str_to_send()

    def get_report_string(self):
        if False:
            print('Hello World!')
        info = self.get_additional_info()
        info['traceback'] = self._get_traceback_str_to_display()
        return self.issue_template.format(**info)

    def get_user_description(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def get_wallet_type(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class EarlyExceptionsQueue:
    """Helper singleton for explicitly sending exceptions to crash reporter.

    Typically the GUIs set up an "exception hook" that catches all otherwise
    uncaught exceptions (which unroll the stack of a thread completely).
    This class provides methods to report *any* exception, and queueing logic
    that delays processing until the exception hook is set up.
    """
    _is_exc_hook_ready = False
    _exc_queue = queue.Queue()

    @classmethod
    def set_hook_as_ready(cls):
        if False:
            return 10
        'Flush the queue and disable it for future exceptions.'
        if cls._is_exc_hook_ready:
            return
        cls._is_exc_hook_ready = True
        while cls._exc_queue.qsize() > 0:
            e = cls._exc_queue.get()
            cls._send_exception_to_crash_reporter(e)

    @classmethod
    def send_exception_to_crash_reporter(cls, e: BaseException):
        if False:
            print('Hello World!')
        if cls._is_exc_hook_ready:
            cls._send_exception_to_crash_reporter(e)
        else:
            cls._exc_queue.put(e)

    @staticmethod
    def _send_exception_to_crash_reporter(e: BaseException):
        if False:
            while True:
                i = 10
        assert EarlyExceptionsQueue._is_exc_hook_ready
        sys.excepthook(type(e), e, e.__traceback__)
send_exception_to_crash_reporter = EarlyExceptionsQueue.send_exception_to_crash_reporter

def trigger_crash():
    if False:
        i = 10
        return i + 15

    class TestingException(Exception):
        pass

    def crash_test():
        if False:
            print('Hello World!')
        raise TestingException('triggered crash for testing purposes')
    import threading
    t = threading.Thread(target=crash_test)
    t.start()
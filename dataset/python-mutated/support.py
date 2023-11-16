import dataclasses
import functools
import math
import os
import pdb
import re
import sys
import time
import traceback
import urllib
from dataclasses import dataclass
import py
import pytest
import toml
from playwright.sync_api import Error as PlaywrightError
ROOT = py.path.local(__file__).dirpath('..', '..', '..')
BUILD = ROOT.join('pyscript.core').join('dist')

def params_with_marks(params):
    if False:
        i = 10
        return i + 15
    "\n    Small helper to automatically apply to each param a pytest.mark with the\n    same name of the param itself. E.g.:\n\n        params_with_marks(['aaa', 'bbb'])\n\n    is equivalent to:\n\n        [pytest.param('aaa', marks=pytest.mark.aaa),\n         pytest.param('bbb', marks=pytest.mark.bbb)]\n\n    This makes it possible to use 'pytest -m aaa' to run ONLY the tests which\n    uses the param 'aaa'.\n    "
    return [pytest.param(name, marks=getattr(pytest.mark, name)) for name in params]

def with_execution_thread(*values):
    if False:
        i = 10
        return i + 15
    "\n    Class decorator to override config.execution_thread.\n\n    By default, we run each test twice:\n      - execution_thread = 'main'\n      - execution_thread = 'worker'\n\n    If you want to execute certain tests with only one specific values of\n    execution_thread, you can use this class decorator. For example:\n\n    @with_execution_thread('main')\n    class TestOnlyMainThread:\n        ...\n\n    @with_execution_thread('worker')\n    class TestOnlyWorker:\n        ...\n\n    If you use @with_execution_thread(None), the logic to inject the\n    execution_thread config is disabled.\n    "
    if values == (None,):

        @pytest.fixture
        def execution_thread(self, request):
            if False:
                for i in range(10):
                    print('nop')
            return None
    else:
        for value in values:
            assert value in ('main', 'worker')

        @pytest.fixture(params=params_with_marks(values))
        def execution_thread(self, request):
            if False:
                for i in range(10):
                    print('nop')
            return request.param

    def with_execution_thread_decorator(cls):
        if False:
            print('Hello World!')
        cls.execution_thread = execution_thread
        return cls
    return with_execution_thread_decorator

def skip_worker(reason):
    if False:
        print('Hello World!')
    "\n    Decorator to skip a test if self.execution_thread == 'worker'\n    "
    if callable(reason):
        raise Exception("You need to specify a reason for skipping, please use: @skip_worker('...')")

    def decorator(fn):
        if False:
            print('Hello World!')

        @functools.wraps(fn)
        def decorated(self, *args):
            if False:
                while True:
                    i = 10
            if self.execution_thread == 'worker':
                pytest.skip(reason)
            return fn(self, *args)
        return decorated
    return decorator

def only_main(fn):
    if False:
        while True:
            i = 10
    '\n    Decorator to mark a test which make sense only in the main thread\n    '

    @functools.wraps(fn)
    def decorated(self, *args):
        if False:
            return 10
        if self.execution_thread == 'worker':
            return
        return fn(self, *args)
    return decorated

def only_worker(fn):
    if False:
        print('Hello World!')
    '\n    Decorator to mark a test which make sense only in the worker thread\n    '

    @functools.wraps(fn)
    def decorated(self, *args):
        if False:
            i = 10
            return i + 15
        if self.execution_thread != 'worker':
            return
        return fn(self, *args)
    return decorated

def filter_inner_text(text, exclude=None):
    if False:
        while True:
            i = 10
    return '\n'.join(filter_page_content(text.splitlines(), exclude=exclude))

def filter_page_content(lines, exclude=None):
    if False:
        for i in range(10):
            print('nop')
    'Remove lines that are not relevant for the test. By default, ignores:\n        (\'\', \'execution_thread = "main"\', \'execution_thread = "worker"\')\n\n    Args:\n        lines (list): list of strings\n        exclude (list): list of strings to exclude\n\n    Returns:\n        list: list of strings\n    '
    if exclude is None:
        exclude = {'', 'execution_thread = "main"', 'execution_thread = "worker"'}
    return [line for line in lines if line not in exclude]

@pytest.mark.usefixtures('init')
@with_execution_thread('main', 'worker')
class PyScriptTest:
    """
    Base class to write PyScript integration tests, based on playwright.

    It provides a simple API to generate HTML files and load them in
    playwright.

    It also provides a Pythonic API on top of playwright for the most
    common tasks; in particular:

      - self.console collects all the JS console.* messages. Look at the doc
        of ConsoleMessageCollection for more details.

      - self.check_js_errors() checks that no JS errors have been thrown

      - after each test, self.check_js_errors() is automatically run to ensure
        that no JS error passes uncaught.

      - self.wait_for_console waits until the specified message appears in the
        console

      - self.wait_for_pyscript waits until all the PyScript tags have been
        evaluated

      - self.pyscript_run is the main entry point for pyscript tests: it
        creates an HTML page to run the specified snippet.
    """
    DEFAULT_TIMEOUT = 30 * 1000

    @pytest.fixture()
    def init(self, request, tmpdir, logger, page, execution_thread):
        if False:
            i = 10
            return i + 15
        '\n        Fixture to automatically initialize all the tests in this class and its\n        subclasses.\n\n        The magic is done by the decorator @pytest.mark.usefixtures("init"),\n        which tells pytest to automatically use this fixture for all the test\n        method of this class.\n\n        Using the standard pytest behavior, we can request more fixtures:\n        tmpdir, and page; \'page\' is a fixture provided by pytest-playwright.\n\n        Then, we save these fixtures on the self and proceed with more\n        initialization. The end result is that the requested fixtures are\n        automatically made available as self.xxx in all methods.\n        '
        self.testname = request.function.__name__.replace('test_', '')
        self.tmpdir = tmpdir
        tmpdir.join('build').mksymlinkto(BUILD)
        self.tmpdir.chdir()
        self.tmpdir.join('favicon.ico').write('')
        self.logger = logger
        self.execution_thread = execution_thread
        self.dev_server = None
        if request.config.option.no_fake_server:
            self.dev_server = request.getfixturevalue('dev_server')
            self.http_server_addr = self.dev_server.base_url
            self.router = None
        else:
            self.http_server_addr = 'https://fake_server'
            self.router = SmartRouter('fake_server', cache=request.config.cache, logger=logger, usepdb=request.config.option.usepdb)
            self.router.install(page)
        self.init_page(page)
        print()
        yield
        if request.config.option.headed:
            pdb.Pdb.intro = '\nThis (Pdb) was started automatically because you passed --headed:\nthe execution of the test pauses here to give you the time to inspect\nthe browser. When you are done, type one of the following commands:\n    (Pdb) continue\n    (Pdb) cont\n    (Pdb) c\n'
            pdb.set_trace()

    def init_page(self, page):
        if False:
            return 10
        self.page = page
        page.set_default_timeout(self.DEFAULT_TIMEOUT)
        self.console = ConsoleMessageCollection(self.logger)
        self._js_errors = []
        self._py_errors = []
        page.on('console', self._on_console)
        page.on('pageerror', self._on_pageerror)

    @property
    def headers(self):
        if False:
            i = 10
            return i + 15
        if self.dev_server is None:
            return self.router.headers
        return self.dev_server.RequestHandlerClass.my_headers()

    def disable_cors_headers(self):
        if False:
            return 10
        if self.dev_server is None:
            self.router.enable_cors_headers = False
        else:
            self.dev_server.RequestHandlerClass.enable_cors_headers = False

    def run_js(self, code):
        if False:
            while True:
                i = 10
        '\n        allows top level await to be present in the `code` parameter\n        '
        self.page.evaluate('(async () => {\n            try {%s}\n            catch(e) {\n                console.error(e);\n            }\n            })();' % code)

    def teardown_method(self):
        if False:
            return 10
        self.check_js_errors()
        self.check_py_errors()

    def _on_console(self, msg):
        if False:
            print('Hello World!')
        if msg.type == 'error' and 'Traceback (most recent call last)' in msg.text:
            self._py_errors.append(msg.text)
        self.console.add_message(msg.type, msg.text)

    def _on_pageerror(self, error):
        if False:
            i = 10
            return i + 15
        error_msg = error.stack or str(error)
        self.console.add_message('js_error', error_msg)
        self._js_errors.append(error_msg)

    def _check_page_errors(self, kind, expected_messages):
        if False:
            return 10
        "\n        Check whether the page raised any 'JS' or 'Python' error.\n\n        expected_messages is a list of strings of errors that you expect they\n        were raised in the page.  They are checked using a simple 'in' check,\n        equivalent to this:\n            if expected_message in actual_error_message:\n                ...\n\n        If an error was expected but not found, it raises PageErrorsDidNotRaise.\n\n        If there are MORE errors other than the expected ones, it raises PageErrors.\n\n        Upon return, all the errors are cleared, so a subsequent call to\n        check_{js,py}_errors will not raise, unless NEW errors have been reported\n        in the meantime.\n        "
        assert kind in ('JS', 'Python')
        if kind == 'JS':
            actual_errors = self._js_errors[:]
        else:
            actual_errors = self._py_errors[:]
        expected_messages = list(expected_messages)
        for (i, msg) in enumerate(expected_messages):
            for (j, error) in enumerate(actual_errors):
                if msg is not None and error is not None and (msg in error):
                    expected_messages[i] = None
                    actual_errors[j] = None
        not_found = [msg for msg in expected_messages if msg is not None]
        unexpected = [err for err in actual_errors if err is not None]
        if kind == 'JS':
            self.clear_js_errors()
        else:
            self.clear_py_errors()
        if not_found:
            raise PageErrorsDidNotRaise(kind, not_found, unexpected)
        if unexpected:
            raise PageErrors(kind, unexpected)

    def check_js_errors(self, *expected_messages):
        if False:
            print('Hello World!')
        '\n        Check whether JS errors were reported.\n\n        See the docstring for _check_page_errors for more details.\n        '
        self._check_page_errors('JS', expected_messages)

    def check_py_errors(self, *expected_messages):
        if False:
            i = 10
            return i + 15
        '\n        Check whether Python errors were reported.\n\n        See the docstring for _check_page_errors for more details.\n        '
        self._check_page_errors('Python', expected_messages)

    def clear_js_errors(self):
        if False:
            return 10
        '\n        Clear all JS errors.\n        '
        self._js_errors = []

    def clear_py_errors(self):
        if False:
            for i in range(10):
                print('nop')
        self._py_errors = []

    def writefile(self, filename, content):
        if False:
            for i in range(10):
                print('nop')
        '\n        Very thin helper to write a file in the tmpdir\n        '
        f = self.tmpdir.join(filename)
        f.dirpath().ensure(dir=True)
        f.write(content)

    def goto(self, path):
        if False:
            for i in range(10):
                print('nop')
        self.logger.reset()
        self.logger.log('page.goto', path, color='yellow')
        url = f'{self.http_server_addr}/{path}'
        self.page.goto(url, timeout=0)

    def wait_for_console(self, text, *, match_substring=False, timeout=None, check_js_errors=True):
        if False:
            return 10
        '\n        Wait until the given message appear in the console. If the message was\n        already printed in the console, return immediately.\n\n        By default "text" must be the *exact* string as printed by a single\n        call to e.g. console.log. If match_substring is True, it is enough\n        that the console contains the given text anywhere.\n\n        timeout is expressed in milliseconds. If it\'s None, it will use\n        the same default as playwright, which is 30 seconds.\n\n        If check_js_errors is True (the default), it also checks that no JS\n        errors were raised during the waiting.\n\n        Return the elapsed time in ms.\n        '
        if match_substring:

            def find_text():
                if False:
                    while True:
                        i = 10
                return text in self.console.all.text
        else:

            def find_text():
                if False:
                    while True:
                        i = 10
                return text in self.console.all.lines
        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT
        try:
            t0 = time.time()
            while True:
                elapsed_ms = (time.time() - t0) * 1000
                if elapsed_ms > timeout:
                    raise TimeoutError(f'{elapsed_ms:.2f} ms')
                if find_text():
                    return elapsed_ms
                self.page.wait_for_timeout(50)
        finally:
            if check_js_errors:
                self.check_js_errors()

    def wait_for_pyscript(self, *, timeout=None, check_js_errors=True):
        if False:
            return 10
        "\n        Wait until pyscript has been fully loaded.\n\n        Timeout is expressed in milliseconds. If it's None, it will use\n        playwright's own default value, which is 30 seconds).\n\n        If check_js_errors is True (the default), it also checks that no JS\n        errors were raised during the waiting.\n        "
        scripts = self.page.locator('script[type=py]').all() + self.page.locator('py-script').all()
        n_scripts = len(scripts)
        elapsed_ms = self.wait_for_console('---py:all-done---', timeout=timeout, check_js_errors=check_js_errors)
        self.logger.log('wait_for_pyscript', f'Waited for {elapsed_ms / 1000:.2f} s', color='yellow')
        self.page.wait_for_selector('html.all-done')
    SCRIPT_TAG_REGEX = re.compile('(<script type="py"|<py-script)')

    def _pyscript_format(self, snippet, *, execution_thread, extra_head=''):
        if False:
            print('Hello World!')
        if execution_thread == 'worker':
            snippet = self.SCRIPT_TAG_REGEX.sub('\\1 worker', snippet)
        doc = f'''\n        <html>\n          <head>\n              <link rel="stylesheet" href="{self.http_server_addr}/build/core.css">\n              <script type="module">\n                import {{ config }} from "{self.http_server_addr}/build/core.js";\n                globalThis.pyConfig = config.py;\n                globalThis.mpyConfig = config.mpy;\n                addEventListener(\n                  'py:all-done',\n                  () => {{\n                    console.debug('---py:all-done---');\n                    document.documentElement.classList.add('all-done');\n                  }},\n                  {{ once: true }}\n                );\n              </script>\n\n              {extra_head}\n          </head>\n          <body>\n            {snippet}\n          </body>\n        </html>\n        '''
        return doc

    def pyscript_run(self, snippet, *, extra_head='', wait_for_pyscript=True, timeout=None, check_js_errors=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Main entry point for pyscript tests.\n\n        snippet contains a fragment of HTML which will be put inside a full\n        HTML document. In particular, the <head> automatically contains the\n        correct <script> and <link> tags which are necessary to load pyscript\n        correctly.\n\n        This method does the following:\n          - write a full HTML file containing the snippet\n          - open a playwright page for it\n          - wait until pyscript has been fully loaded\n        '
        doc = self._pyscript_format(snippet, execution_thread=self.execution_thread, extra_head=extra_head)
        if not wait_for_pyscript and timeout is not None:
            raise ValueError('Cannot set a timeout if wait_for_pyscript=False')
        filename = f'{self.testname}.html'
        self.writefile(filename, doc)
        self.goto(filename)
        if wait_for_pyscript:
            self.wait_for_pyscript(timeout=timeout, check_js_errors=check_js_errors)

    def iter_locator(self, loc):
        if False:
            i = 10
            return i + 15
        '\n        Helper method to iterate over all the elements which are matched by a\n        locator, since playwright does not seem to support it natively.\n        '
        n = loc.count()
        elems = [loc.nth(i) for i in range(n)]
        return iter(elems)

    def assert_no_banners(self):
        if False:
            return 10
        '\n        Ensure that there are no alert banners on the page, which are used for\n        errors and warnings. Raise AssertionError if any if found.\n        '
        loc = self.page.locator('.alert-banner')
        n = loc.count()
        if n > 0:
            text = '\n'.join(loc.all_inner_texts())
            raise AssertionError(f'Found {n} alert banners:\n' + text)

    def assert_banner_message(self, expected_message):
        if False:
            while True:
                i = 10
        '\n        Ensure that there is an alert banner on the page with the given message.\n        Currently it only handles a single.\n        '
        banner = self.page.wait_for_selector('.py-error')
        banner_text = banner.inner_text()
        if expected_message not in banner_text:
            raise AssertionError(f"Expected message '{expected_message}' does not match banner text '{banner_text}'")
        return True

    def check_tutor_generated_code(self, modules_to_check=None):
        if False:
            while True:
                i = 10
        '\n        Ensure that the source code viewer injected by the PyTutor plugin\n        is presend. Raise AssertionError if not found.\n\n        Args:\n\n            modules_to_check(str): iterable with names of the python modules\n                                that have been included in the tutor config\n                                and needs to be checked (if they are included\n                                in the displayed source code)\n\n        Returns:\n            None\n        '
        assert self.page.locator('py-tutor').count()
        view_code_button = self.page.locator('#view-code-button')
        vcb_count = view_code_button.count()
        if vcb_count != 1:
            raise AssertionError(f'Found {vcb_count} code view button. Should have been 1!')
        code_section = self.page.locator('#code-section')
        code_section_count = code_section.count()
        code_msg = f'One (and only one) code section should exist. Found: {code_section_count}'
        assert code_section_count == 1, code_msg
        pyconfig_tag = self.page.locator('py-config')
        code_section_inner_html = code_section.inner_html()
        assert '<p>index.html</p>' in code_section_inner_html
        assert '<pre class="prism-code language-html" tabindex="0">    <code class="language-html">' in code_section_inner_html
        if modules_to_check:
            for module in modules_to_check:
                assert f'{module}' in code_section_inner_html
        assert '&lt;</span>py-config</span>' in code_section_inner_html
        assert pyconfig_tag.inner_html() in code_section_inner_html
        assert 'code-section-hidden' in code_section.get_attribute('class')
        view_code_button.click()
        assert 'code-section-visible' in code_section.get_attribute('class')
MAX_TEST_TIME = 30
TEST_TIME_INCREMENT = 0.25
TEST_ITERATIONS = math.ceil(MAX_TEST_TIME / TEST_TIME_INCREMENT)

def wait_for_render(page, selector, pattern, timeout_seconds=None):
    if False:
        while True:
            i = 10
    '\n    Assert that rendering inserts data into the page as expected: search the\n    DOM from within the timing loop for a string that is not present in the\n    initial markup but should appear by way of rendering\n    '
    re_sub_content = re.compile(pattern)
    py_rendered = False
    if timeout_seconds:
        check_iterations = math.ceil(timeout_seconds / TEST_TIME_INCREMENT)
    else:
        check_iterations = TEST_ITERATIONS
    for _ in range(check_iterations):
        content = page.inner_html(selector)
        if re_sub_content.search(content):
            py_rendered = True
            break
        time.sleep(TEST_TIME_INCREMENT)
    assert py_rendered

class PageErrors(Exception):
    """
    Represent one or more exceptions which happened in JS or Python.
    """

    def __init__(self, kind, errors):
        if False:
            for i in range(10):
                print('nop')
        assert kind in ('JS', 'Python')
        n = len(errors)
        assert n != 0
        lines = [f'{kind} errors found: {n}']
        lines += errors
        msg = '\n'.join(lines)
        super().__init__(msg)
        self.errors = errors

class PageErrorsDidNotRaise(Exception):
    """
    Exception raised by check_{js,py}_errors when the expected JS or Python
    error messages are not found.
    """

    def __init__(self, kind, expected_messages, errors):
        if False:
            print('Hello World!')
        assert kind in ('JS', 'Python')
        lines = [f'The following {kind} errors were expected but could not be found:']
        for msg in expected_messages:
            lines.append('    - ' + msg)
        if errors:
            lines.append('---')
            lines.append(f'The following {kind} errors were raised but not expected:')
            lines += errors
        msg = '\n'.join(lines)
        super().__init__(msg)
        self.expected_messages = expected_messages
        self.errors = errors

class ConsoleMessageCollection:
    """
    Helper class to collect and expose ConsoleMessage in a Pythonic way.

    Usage:

      console.log.messages: list of ConsoleMessage with type=='log'
      console.log.lines:    list of strings
      console.log.text:     the whole text as single string

      console.debug.*       same as above, but with different types
      console.info.*
      console.error.*
      console.warning.*

      console.js_error.*    this is a special category which does not exist in the
                            browser: it prints uncaught JS exceptions

      console.all.*         same as the individual categories but considering
                            all messages which were sent to the console
    """

    @dataclass
    class Message:
        type: str
        text: str

    class View:
        """
        Filter console messages by the given msg_type
        """

        def __init__(self, console, msg_type):
            if False:
                for i in range(10):
                    print('nop')
            self.console = console
            self.msg_type = msg_type

        @property
        def messages(self):
            if False:
                print('Hello World!')
            if self.msg_type is None:
                return self.console._messages
            else:
                return [msg for msg in self.console._messages if msg.type == self.msg_type]

        @property
        def lines(self):
            if False:
                return 10
            return [msg.text for msg in self.messages]

        @property
        def text(self):
            if False:
                i = 10
                return i + 15
            return '\n'.join(self.lines)
    _COLORS = {'warning': 'brown', 'error': 'darkred', 'js_error': 'red'}

    def __init__(self, logger):
        if False:
            print('Hello World!')
        self.logger = logger
        self._messages = []
        self.all = self.View(self, None)
        self.log = self.View(self, 'log')
        self.debug = self.View(self, 'debug')
        self.info = self.View(self, 'info')
        self.error = self.View(self, 'error')
        self.warning = self.View(self, 'warning')
        self.js_error = self.View(self, 'js_error')

    def add_message(self, type, text):
        if False:
            while True:
                i = 10
        msg = self.Message(type=type, text=text)
        category = f'console.{msg.type}'
        color = self._COLORS.get(msg.type)
        self.logger.log(category, msg.text, color=color)
        self._messages.append(msg)

class Logger:
    """
    Helper class to log messages to stdout.

    Features:
      - nice formatted category
      - keep track of time passed since the last reset
      - support colors

    NOTE: the (lowercase) logger fixture is defined in conftest.py
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.reset()
        self.prefix_regexp = re.compile('(\\[.+?\\])')

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.start_time = time.time()

    def colorize_prefix(self, text, *, color):
        if False:
            while True:
                i = 10
        (start, end) = Color.escape_pair(color)
        return self.prefix_regexp.sub(f'{start}\\1{end}', text, 1)

    def log(self, category, text, *, color=None):
        if False:
            i = 10
            return i + 15
        delta = time.time() - self.start_time
        text = self.colorize_prefix(text, color='teal')
        line = f'[{delta:6.2f} {category:17}] {text}'
        if color:
            line = Color.set(color, line)
        print(line)

class Color:
    """
    Helper method to print colored output using ANSI escape codes.
    """
    black = '30'
    darkred = '31'
    darkgreen = '32'
    brown = '33'
    darkblue = '34'
    purple = '35'
    teal = '36'
    lightgray = '37'
    darkgray = '30;01'
    red = '31;01'
    green = '32;01'
    yellow = '33;01'
    blue = '34;01'
    fuchsia = '35;01'
    turquoise = '36;01'
    white = '37;01'

    @classmethod
    def set(cls, color, string):
        if False:
            while True:
                i = 10
        (start, end) = cls.escape_pair(color)
        return f'{start}{string}{end}'

    @classmethod
    def escape_pair(cls, color):
        if False:
            i = 10
            return i + 15
        try:
            color = getattr(cls, color)
        except AttributeError:
            pass
        start = f'\x1b[{color}m'
        end = '\x1b[00m'
        return (start, end)

class SmartRouter:
    """
    A smart router to be used in conjunction with playwright.Page.route.

    Main features:

      - it intercepts the requests to a local "fake server" and serve them
        statically from disk

      - it intercepts the requests to the network and cache the results
        locally
    """

    @dataclass
    class CachedResponse:
        """
        We cannot put playwright's APIResponse instances inside _cache, because
        they are valid only in the context of the same page. As a workaround,
        we manually save status, headers and body of each cached response.
        """
        status: int
        headers: dict
        body: str

        def asdict(self):
            if False:
                while True:
                    i = 10
            return dataclasses.asdict(self)

        @classmethod
        def fromdict(cls, d):
            if False:
                i = 10
                return i + 15
            return cls(**d)

    def __init__(self, fake_server, *, cache, logger, usepdb=False):
        if False:
            return 10
        '\n        fake_server: the domain name of the fake server\n        '
        self.fake_server = fake_server
        self.cache = cache
        self.logger = logger
        self.usepdb = usepdb
        self.page = None
        self.requests = []
        self.enable_cors_headers = True

    @property
    def headers(self):
        if False:
            i = 10
            return i + 15
        if self.enable_cors_headers:
            return {'Cross-Origin-Embedder-Policy': 'require-corp', 'Cross-Origin-Opener-Policy': 'same-origin'}
        return {}

    def install(self, page):
        if False:
            return 10
        '\n        Install the smart router on a page\n        '
        self.page = page
        self.page.route('**', self.router)

    def router(self, route):
        if False:
            for i in range(10):
                print('nop')
        "\n        Intercept and fulfill playwright requests.\n\n        NOTE!\n        If we raise an exception inside router, playwright just hangs and the\n        exception seems not to be propagated outside. It's very likely a\n        playwright bug.\n\n        This means that for example pytest doesn't have any chance to\n        intercept the exception and fail in a meaningful way.\n\n        As a workaround, we try to intercept exceptions by ourselves, print\n        something reasonable on the console and abort the request (hoping that\n        the test will fail cleaninly, that's the best we can do). We also try\n        to respect pytest --pdb, for what it's possible.\n        "
        try:
            return self._router(route)
        except Exception:
            print('***** Error inside Fake_Server.router *****')
            info = sys.exc_info()
            print(traceback.format_exc())
            if self.usepdb:
                pdb.post_mortem(info[2])
            route.abort()

    def log_request(self, status, kind, url):
        if False:
            print('Hello World!')
        self.requests.append((status, kind, url))
        color = 'blue' if status == 200 else 'red'
        self.logger.log('request', f'{status} - {kind} - {url}', color=color)

    def _router(self, route):
        if False:
            print('Hello World!')
        full_url = route.request.url
        url = urllib.parse.urlparse(full_url)
        assert url.scheme in ('http', 'https')
        if url.netloc == self.fake_server:
            self.log_request(200, 'fake_server', full_url)
            assert url.path[0] == '/'
            relative_path = url.path[1:]
            if os.path.exists(relative_path):
                route.fulfill(status=200, headers=self.headers, path=relative_path)
            else:
                route.fulfill(status=404, headers=self.headers)
            return
        resp = self.fetch_from_cache(full_url)
        if resp is not None:
            kind = 'CACHED'
        else:
            kind = 'NETWORK'
            resp = self.fetch_from_network(route.request)
            self.save_resp_to_cache(full_url, resp)
        self.log_request(resp.status, kind, full_url)
        route.fulfill(status=resp.status, headers=resp.headers, body=resp.body)

    def clear_cache(self, url):
        if False:
            return 10
        key = 'pyscript/' + url
        self.cache.set(key, None)

    def save_resp_to_cache(self, url, resp):
        if False:
            i = 10
            return i + 15
        key = 'pyscript/' + url
        data = resp.asdict()
        data['body'] = data['body'].decode('latin-1')
        self.cache.set(key, data)

    def fetch_from_cache(self, url):
        if False:
            for i in range(10):
                print('nop')
        key = 'pyscript/' + url
        data = self.cache.get(key, None)
        if data is None:
            return None
        data['body'] = data['body'].encode('latin-1')
        return self.CachedResponse(**data)

    def fetch_from_network(self, request):
        if False:
            return 10
        try:
            api_response = self.page.request.fetch(request)
        except PlaywrightError:
            time.sleep(0.5)
            api_response = self.page.request.fetch(request)
        cached_response = self.CachedResponse(status=api_response.status, headers=api_response.headers, body=api_response.body())
        return cached_response
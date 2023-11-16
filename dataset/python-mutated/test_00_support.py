import re
import textwrap
import pytest
from .support import PageErrors, PageErrorsDidNotRaise, PyScriptTest, with_execution_thread

@with_execution_thread(None)
class TestSupport(PyScriptTest):
    """
    These are NOT tests about PyScript.

    They test the PyScriptTest class, i.e. we want to ensure that all the
    testing machinery that we have works correctly.
    """

    def test_basic(self):
        if False:
            return 10
        '\n        Very basic test, just to check that we can write, serve and read a simple\n        HTML (no pyscript yet)\n        '
        doc = '\n        <html>\n          <body>\n            <h1>Hello world</h1>\n          </body>\n        </html>\n        '
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        content = self.page.content()
        assert '<h1>Hello world</h1>' in content

    def test_await_with_run_js(self):
        if False:
            return 10
        self.run_js('\n          function resolveAfter200MilliSeconds(x) {\n            return new Promise((resolve) => {\n              setTimeout(() => {\n                resolve(x);\n              }, 200);\n            });\n          }\n\n          const x = await resolveAfter200MilliSeconds(10);\n          console.log(x);\n        ')
        assert self.console.log.lines[-1] == '10'

    def test_console(self):
        if False:
            while True:
                i = 10
        '\n        Test that we capture console.log messages correctly.\n        '
        doc = '\n        <html>\n          <body>\n            <script>\n                console.log("my log 1");\n                console.debug("my debug");\n                console.info("my info");\n                console.error("my error");\n                console.warn("my warning");\n                console.log("my log 2");\n            </script>\n          </body>\n        </html>\n        '
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        assert len(self.console.all.messages) == 6
        assert self.console.all.lines == ['my log 1', 'my debug', 'my info', 'my error', 'my warning', 'my log 2']
        assert self.console.all.text == textwrap.dedent('\n            my log 1\n            my debug\n            my info\n            my error\n            my warning\n            my log 2\n        ').strip()
        assert self.console.log.lines == ['my log 1', 'my log 2']
        assert self.console.debug.lines == ['my debug']

    def test_check_js_errors_simple(self):
        if False:
            for i in range(10):
                print('nop')
        doc = "\n        <html>\n          <body>\n            <script>throw new Error('this is an error');</script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        with pytest.raises(PageErrors) as exc:
            self.check_js_errors()
        msg = str(exc.value)
        expected = textwrap.dedent(f'\n            JS errors found: 1\n            Error: this is an error\n                at {self.http_server_addr}/mytest.html:.*\n            ').strip()
        assert re.search(expected, msg)
        self.check_js_errors()
        assert self.console.js_error.lines[0].startswith('Error: this is an error')

    def test_check_js_errors_expected(self):
        if False:
            i = 10
            return i + 15
        doc = "\n        <html>\n          <body>\n            <script>throw new Error('this is an error');</script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        self.check_js_errors('this is an error')

    def test_check_js_errors_expected_but_didnt_raise(self):
        if False:
            i = 10
            return i + 15
        doc = "\n        <html>\n          <body>\n            <script>throw new Error('this is an error 2');</script>\n            <script>throw new Error('this is an error 4');</script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        with pytest.raises(PageErrorsDidNotRaise) as exc:
            self.check_js_errors('this is an error 1', 'this is an error 2', 'this is an error 3', 'this is an error 4')
        msg = str(exc.value)
        expected = textwrap.dedent('\n            The following JS errors were expected but could not be found:\n                - this is an error 1\n                - this is an error 3\n            ').strip()
        assert re.search(expected, msg)

    def test_check_js_errors_multiple(self):
        if False:
            i = 10
            return i + 15
        doc = "\n        <html>\n          <body>\n            <script>throw new Error('error 1');</script>\n            <script>throw new Error('error 2');</script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        with pytest.raises(PageErrors) as exc:
            self.check_js_errors()
        msg = str(exc.value)
        expected = textwrap.dedent('\n            JS errors found: 2\n            Error: error 1\n                at https://fake_server/mytest.html:.*\n            Error: error 2\n                at https://fake_server/mytest.html:.*\n            ').strip()
        assert re.search(expected, msg)
        self.check_js_errors()

    def test_check_js_errors_some_expected_but_others_not(self):
        if False:
            for i in range(10):
                print('nop')
        doc = "\n        <html>\n          <body>\n            <script>throw new Error('expected 1');</script>\n            <script>throw new Error('NOT expected 2');</script>\n            <script>throw new Error('expected 3');</script>\n            <script>throw new Error('NOT expected 4');</script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        with pytest.raises(PageErrors) as exc:
            self.check_js_errors('expected 1', 'expected 3')
        msg = str(exc.value)
        expected = textwrap.dedent('\n            JS errors found: 2\n            Error: NOT expected 2\n                at https://fake_server/mytest.html:.*\n            Error: NOT expected 4\n                at https://fake_server/mytest.html:.*\n            ').strip()
        assert re.search(expected, msg)

    def test_check_js_errors_expected_not_found_but_other_errors(self):
        if False:
            i = 10
            return i + 15
        doc = "\n        <html>\n          <body>\n            <script>throw new Error('error 1');</script>\n            <script>throw new Error('error 2');</script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        with pytest.raises(PageErrorsDidNotRaise) as exc:
            self.check_js_errors('this is not going to be found')
        msg = str(exc.value)
        expected = textwrap.dedent('\n            The following JS errors were expected but could not be found:\n                - this is not going to be found\n            ---\n            The following JS errors were raised but not expected:\n            Error: error 1\n                at https://fake_server/mytest.html:.*\n            Error: error 2\n                at https://fake_server/mytest.html:.*\n            ').strip()
        assert re.search(expected, msg)

    def test_clear_js_errors(self):
        if False:
            return 10
        doc = "\n        <html>\n          <body>\n            <script>throw new Error('this is an error');</script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        self.clear_js_errors()
        self.check_js_errors()

    def test_wait_for_console_simple(self):
        if False:
            print('Hello World!')
        "\n        Test that self.wait_for_console actually waits.\n        If it's buggy, the test will try to read self.console.log BEFORE the\n        log has been written and it will fail.\n        "
        doc = "\n        <html>\n          <body>\n            <script>\n                setTimeout(function() {\n                    console.log('Page loaded!');\n                }, 100);\n            </script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        self.wait_for_console('Page loaded!', timeout=200)
        assert self.console.log.lines[-1] == 'Page loaded!'

    def test_wait_for_console_timeout(self):
        if False:
            return 10
        doc = '\n        <html>\n          <body>\n          </body>\n        </html>\n        '
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        with pytest.raises(TimeoutError):
            self.wait_for_console('This text will never be printed', timeout=200)

    def test_wait_for_console_dont_wait_if_already_emitted(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the text is already on the console, wait_for_console() should return\n        immediately without waiting.\n        '
        doc = "\n        <html>\n          <body>\n            <script>\n                console.log('Hello world')\n                console.log('Page loaded!');\n            </script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        self.wait_for_console('Page loaded!', timeout=200)
        assert self.console.log.lines[-2] == 'Hello world'
        assert self.console.log.lines[-1] == 'Page loaded!'
        self.wait_for_console('Hello world', timeout=1)

    def test_wait_for_console_exception_1(self):
        if False:
            return 10
        '\n        Test that if a JS exception is raised while waiting for the console, we\n        report the exception and not the timeout.\n\n        There are two main cases:\n           1. there is an exception and the console message does not appear\n           2. there is an exception but the console message appears anyway\n\n        This test checks for case 1. Case 2 is tested by\n        test_wait_for_console_exception_2\n        '
        doc = "\n        <html>\n          <body>\n            <script>throw new Error('this is an error');</script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        with pytest.raises(PageErrors) as exc:
            self.wait_for_console('Page loaded!', timeout=200)
        assert 'this is an error' in str(exc.value)
        assert isinstance(exc.value.__context__, TimeoutError)
        self.goto('mytest.html')
        with pytest.raises(TimeoutError):
            self.wait_for_console('Page loaded!', timeout=200, check_js_errors=False)
        self.clear_js_errors()

    def test_wait_for_console_exception_2(self):
        if False:
            return 10
        '\n        See the description in test_wait_for_console_exception_1.\n        '
        doc = "\n        <html>\n          <body>\n            <script>\n                setTimeout(function() {\n                    console.log('Page loaded!');\n                }, 100);\n                throw new Error('this is an error');\n            </script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        with pytest.raises(PageErrors) as exc:
            self.wait_for_console('Page loaded!', timeout=200)
        assert 'this is an error' in str(exc.value)
        self.goto('mytest.html')
        self.wait_for_console('Page loaded!', timeout=200, check_js_errors=False)
        self.clear_js_errors()

    def test_wait_for_console_match_substring(self):
        if False:
            for i in range(10):
                print('nop')
        doc = "\n        <html>\n          <body>\n            <script>\n                console.log('Foo Bar Baz');\n            </script>\n          </body>\n        </html>\n        "
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        with pytest.raises(TimeoutError):
            self.wait_for_console('Bar', timeout=200)
        self.wait_for_console('Bar', timeout=200, match_substring=True)
        assert self.console.log.lines[-1] == 'Foo Bar Baz'

    def test_iter_locator(self):
        if False:
            for i in range(10):
                print('nop')
        doc = '\n        <html>\n          <body>\n              <div>foo</div>\n              <div>bar</div>\n              <div>baz</div>\n          </body>\n        </html>\n        '
        self.writefile('mytest.html', doc)
        self.goto('mytest.html')
        divs = self.page.locator('div')
        assert divs.count() == 3
        texts = [el.inner_text() for el in self.iter_locator(divs)]
        assert texts == ['foo', 'bar', 'baz']

    def test_smartrouter_cache(self):
        if False:
            return 10
        if self.router is None:
            pytest.skip('Cannot test SmartRouter with --dev')
        URL = 'https://raw.githubusercontent.com/pyscript/pyscript/main/README.md'
        doc = f'\n        <html>\n          <body>\n              <img src="{URL}">\n          </body>\n        </html>\n        '
        self.writefile('mytest.html', doc)
        self.router.clear_cache(URL)
        self.goto('mytest.html')
        assert self.router.requests == [(200, 'fake_server', 'https://fake_server/mytest.html'), (200, 'NETWORK', URL)]
        self.goto('mytest.html')
        assert self.router.requests == [(200, 'fake_server', 'https://fake_server/mytest.html'), (200, 'NETWORK', URL), (200, 'fake_server', 'https://fake_server/mytest.html'), (200, 'CACHED', URL)]

    def test_404(self):
        if False:
            while True:
                i = 10
        '\n        Test that we capture a 404 in loading a page that does not exist.\n        '
        self.goto('this_url_does_not_exist.html')
        assert ['Failed to load resource: the server responded with a status of 404 (Not Found)'] == self.console.all.lines
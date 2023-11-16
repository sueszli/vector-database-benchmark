import re
import pytest
from .support import PyScriptTest, only_main, skip_worker

class TestBasic(PyScriptTest):

    def test_pyscript_exports(self):
        if False:
            return 10
        self.pyscript_run('\n            <script type="py">\n                from pyscript import RUNNING_IN_WORKER, PyWorker, window, document, sync, current_target\n            </script>\n            ')
        assert self.console.error.lines == []

    def test_script_py_hello(self):
        if False:
            return 10
        self.pyscript_run('\n            <script type="py">\n                import js\n                js.console.log(\'hello from script py\')\n            </script>\n            ')
        assert self.console.log.lines == ['hello from script py']

    def test_py_script_hello(self):
        if False:
            while True:
                i = 10
        self.pyscript_run("\n            <py-script>\n                import js\n                js.console.log('hello from py-script')\n            </py-script>\n            ")
        assert self.console.log.lines == ['hello from py-script']

    def test_execution_thread(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="py">\n                import pyscript\n                import js\n                js.console.log("worker?", pyscript.RUNNING_IN_WORKER)\n            </script>\n            ')
        assert self.execution_thread in ('main', 'worker')
        in_worker = self.execution_thread == 'worker'
        in_worker = str(in_worker).lower()
        assert self.console.log.lines[-1] == f'worker? {in_worker}'

    @skip_worker('NEXT: it should show a nice error on the page')
    def test_no_cors_headers(self):
        if False:
            while True:
                i = 10
        self.disable_cors_headers()
        self.pyscript_run('\n            <script type="py">\n                import js\n                js.console.log("hello")\n            </script>\n            ', wait_for_pyscript=False)
        assert self.headers == {}
        if self.execution_thread == 'main':
            self.wait_for_pyscript()
            assert self.console.log.lines == ['hello']
            self.assert_no_banners()
        else:
            expected_alert_banner_msg = '(PY1000): When execution_thread is "worker", the site must be cross origin isolated, but crossOriginIsolated is false. To be cross origin isolated, the server must use https and also serve with the following headers: {"Cross-Origin-Embedder-Policy":"require-corp","Cross-Origin-Opener-Policy":"same-origin"}. The problem may be that one or both of these are missing.'
            alert_banner = self.page.wait_for_selector('.py-error')
            assert expected_alert_banner_msg in alert_banner.inner_text()

    def test_print(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <script type="py">\n                print(\'hello pyscript\')\n            </script>\n            ')
        assert self.console.log.lines[-1] == 'hello pyscript'

    @only_main
    def test_input_exception(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <script type="py">\n                input("what\'s your name?")\n            </script>\n            ')
        self.check_py_errors("Exception: input() doesn't work when PyScript runs in the main thread.")

    @skip_worker('NEXT: exceptions should be displayed in the DOM')
    def test_python_exception(self):
        if False:
            return 10
        self.pyscript_run('\n            <script type="py">\n                print(\'hello pyscript\')\n                raise Exception(\'this is an error\')\n            </script>\n        ')
        assert 'hello pyscript' in self.console.log.lines
        self.check_py_errors('Exception: this is an error')
        banner = self.page.locator('.py-error')
        tb_lines = banner.inner_text().splitlines()
        assert tb_lines[0] == 'Traceback (most recent call last):'
        assert tb_lines[-1] == 'Exception: this is an error'

    @skip_worker("NEXT: py-click doesn't work inside workers")
    def test_python_exception_in_event_handler(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <button py-click="onclick">Click me</button>\n            <script type="py">\n                def onclick(event):\n                    raise Exception("this is an error inside handler")\n            </script>\n        ')
        self.page.locator('button').click()
        self.wait_for_console('Exception: this is an error inside handler', match_substring=True)
        self.check_py_errors('Exception: this is an error inside handler')
        tb_lines = self.page.locator('.py-error').inner_text().splitlines()
        assert tb_lines[0] == 'Traceback (most recent call last):'
        assert tb_lines[-1] == 'Exception: this is an error inside handler'

    @only_main
    def test_execution_in_order(self):
        if False:
            print('Hello World!')
        '\n        Check that they script py tags are executed in the same order they are\n        defined\n        '
        self.pyscript_run('\n            <script type="py">import js; js.console.log(\'one\')</script>\n            <script type="py">js.console.log(\'two\')</script>\n            <script type="py">js.console.log(\'three\')</script>\n            <script type="py">js.console.log(\'four\')</script>\n        ')
        assert self.console.log.lines[-4:] == ['one', 'two', 'three', 'four']

    def test_escaping_of_angle_brackets(self):
        if False:
            while True:
                i = 10
        '\n        Check that script tags escape angle brackets\n        '
        self.pyscript_run('\n            <script type="py">\n                import js\n                js.console.log("A", 1<2, 1>2)\n                js.console.log("B <div></div>")\n            </script>\n            <py-script>\n                import js\n                js.console.log("C", 1<2, 1>2)\n                js.console.log("D <div></div>")\n            </py-script>\n        ')
        lines = sorted(self.console.log.lines[-4:])
        assert lines == ['A true false', 'B <div></div>', 'C true false', 'D <div></div>']

    def test_packages(self):
        if False:
            return 10
        self.pyscript_run('\n            <py-config>\n                packages = ["asciitree"]\n            </py-config>\n            <script type="py">\n                import js\n                import asciitree\n                js.console.log(\'hello\', asciitree.__name__)\n            </script>\n            ')
        assert self.console.log.lines[-3:] == ['Loading asciitree', 'Loaded asciitree', 'hello asciitree']

    @pytest.mark.skip('NEXT: No banner')
    def test_non_existent_package(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <py-config>\n                packages = ["i-dont-exist"]\n            </py-config>\n            <script type="py">\n                print(\'hello\')\n            </script>\n            ', wait_for_pyscript=False)
        expected_alert_banner_msg = "(PY1001): Unable to install package(s) 'i-dont-exist'. Unable to find package in PyPI. Please make sure you have entered a correct package name."
        alert_banner = self.page.wait_for_selector('.alert-banner')
        assert expected_alert_banner_msg in alert_banner.inner_text()
        self.check_py_errors("Can't fetch metadata for 'i-dont-exist'")

    @pytest.mark.skip('NEXT: No banner')
    def test_no_python_wheel(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <py-config>\n                packages = ["opsdroid"]\n            </py-config>\n            <script type="py">\n                print(\'hello\')\n            </script>\n            ', wait_for_pyscript=False)
        expected_alert_banner_msg = "(PY1001): Unable to install package(s) 'opsdroid'. Reason: Can't find a pure Python 3 Wheel for package(s) 'opsdroid'"
        alert_banner = self.page.wait_for_selector('.alert-banner')
        assert expected_alert_banner_msg in alert_banner.inner_text()
        self.check_py_errors("Can't find a pure Python 3 wheel for 'opsdroid'")

    @only_main
    def test_dynamically_add_py_script_tag(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <script>\n                function addPyScriptTag(event) {\n                    let tag = document.createElement(\'py-script\');\n                    tag.innerHTML = "print(\'hello world\')";\n                    document.body.appendChild(tag);\n                }\n                addPyScriptTag()\n            </script>\n            ', timeout=20000)
        self.page.locator('py-script')
        assert self.console.log.lines[-1] == 'hello world'

    def test_py_script_src_attribute(self):
        if False:
            i = 10
            return i + 15
        self.writefile('foo.py', "print('hello from foo')")
        self.pyscript_run('\n            <script type="py" src="foo.py"></script>\n            ')
        assert self.console.log.lines[-1] == 'hello from foo'

    @skip_worker('NEXT: banner not shown')
    def test_py_script_src_not_found(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="py" src="foo.py"></script>\n            ', check_js_errors=False)
        assert 'Failed to load resource' in self.console.error.lines[0]
        expected_msg = '(PY0404): Fetching from URL foo.py failed with error 404'
        assert any((expected_msg in line for line in self.console.error.lines))
        assert self.assert_banner_message(expected_msg)

    @pytest.mark.skip("NEXT: we don't expose pyscript on window")
    def test_js_version(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <script type="py">\n            </script>\n            ')
        self.page.add_script_tag(content='console.log(pyscript.version)')
        assert re.match('\\d{4}\\.\\d{2}\\.\\d+(\\.[a-zA-Z0-9]+)?', self.console.log.lines[-1]) is not None

    @pytest.mark.skip("NEXT: we don't expose pyscript on window")
    def test_python_version(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n        <script type="py">\n            import js\n            js.console.log(pyscript.__version__)\n            js.console.log(str(pyscript.version_info))\n        </script>\n        ')
        assert re.match('\\d{4}\\.\\d{2}\\.\\d+(\\.[a-zA-Z0-9]+)?', self.console.log.lines[-2]) is not None
        assert re.match("version_info\\(year=\\d{4}, month=\\d{2}, minor=\\d+, releaselevel='([a-zA-Z0-9]+)?'\\)", self.console.log.lines[-1]) is not None

    @pytest.mark.skip('NEXT: works with <py-script> not with <script>')
    def test_getPySrc_returns_source_code(self):
        if False:
            return 10
        self.pyscript_run('\n            <py-script>print("hello from py-script")</py-script>\n            <script type="py">print("hello from script py")</script>\n            ')
        pyscript_tag = self.page.locator('py-script')
        assert pyscript_tag.inner_html() == ''
        assert pyscript_tag.evaluate('node => node.srcCode') == 'print("hello from py-script")'
        script_py_tag = self.page.locator('script[type="py"]')
        assert script_py_tag.evaluate('node => node.srcCode') == 'print("hello from script py")'

    @skip_worker("NEXT: py-click doesn't work inside workers")
    def test_py_attribute_without_id(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <button py-click="myfunc">Click me</button>\n            <script type="py">\n                def myfunc(event):\n                    print("hello world!")\n            </script>\n            ')
        btn = self.page.wait_for_selector('button')
        btn.click()
        self.wait_for_console('hello world!')
        assert self.console.log.lines[-1] == 'hello world!'
        assert self.console.error.lines == []

    def test_py_all_done_event(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script>\n                addEventListener("py:all-done", () => console.log("2"))\n            </script>\n            <script type="py">\n                print("1")\n            </script>\n            ')
        assert self.console.log.lines == ['1', '2']
        assert self.console.error.lines == []
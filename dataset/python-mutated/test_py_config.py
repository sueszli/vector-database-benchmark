import os
import pytest
from .support import PyScriptTest, with_execution_thread

@with_execution_thread(None)
class TestConfig(PyScriptTest):

    def test_py_config_inline_pyscript(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n        <py-config>\n            name = "foobar"\n        </py-config>\n\n        <py-script async>\n            from pyscript import window\n            window.console.log("config name:", window.pyConfig.name)\n        </py-script>\n        ')
        assert self.console.log.lines[-1] == 'config name: foobar'

    @pytest.mark.skip('NEXT: works with <py-script> not with <script>')
    def test_py_config_inline_scriptpy(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n        <py-config>\n            name = "foobar"\n        </py-config>\n\n        <script type="py" async>\n            from pyscript import window\n            window.console.log("config name:", window.pyConfig.name)\n        </script>\n        ')
        assert self.console.log.lines[-1] == 'config name: foobar'

    @pytest.mark.skip('NEXT: works with <py-script> not with <script>')
    def test_py_config_external(self):
        if False:
            while True:
                i = 10
        pyconfig_toml = '\n            name = "app with external config"\n        '
        self.writefile('pyconfig.toml', pyconfig_toml)
        self.pyscript_run('\n        <py-config src="pyconfig.toml"></py-config>\n\n        <script type="py" async>\n            from pyscript import window\n            window.console.log("config name:", window.pyConfig.name)\n        </script>\n        ')
        assert self.console.log.lines[-1] == 'config name: app with external config'

    def test_invalid_json_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <py-config type="json">\n                [[\n            </py-config>\n            ', wait_for_pyscript=False)
        banner = self.page.wait_for_selector('.py-error')
        expected = '(PY1000): Invalid JSON\nUnexpected end of JSON input'
        assert banner.inner_text() == expected

    def test_invalid_toml_config(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <py-config>\n                [[\n            </py-config>\n            ', wait_for_pyscript=False)
        banner = self.page.wait_for_selector('.py-error')
        expected = '(PY1000): Invalid TOML\nExpected DoubleQuote, Whitespace, or [a-z], [A-Z], [0-9], "-", "_" but end of input found.'
        assert banner.inner_text() == expected

    @pytest.mark.skip('NEXT: emit a warning in case of multiple py-config')
    def test_multiple_py_config(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <py-config>\n            name = "foobar"\n            </py-config>\n\n            <py-config>\n            name = "this is ignored"\n            </py-config>\n\n            <script type="py">\n                import js\n                #config = js.pyscript_get_config()\n                #js.console.log("config name:", config.name)\n            </script>\n            ')
        banner = self.page.wait_for_selector('.py-warning')
        expected = 'Multiple <py-config> tags detected. Only the first is going to be parsed, all the others will be ignored'
        assert banner.text_content() == expected

    def test_paths(self):
        if False:
            for i in range(10):
                print('nop')
        self.writefile('a.py', "x = 'hello from A'")
        self.writefile('b.py', "x = 'hello from B'")
        self.pyscript_run('\n            <py-config>\n                [[fetch]]\n                files = ["./a.py", "./b.py"]\n            </py-config>\n\n            <script type="py">\n                import js\n                import a, b\n                js.console.log(a.x)\n                js.console.log(b.x)\n            </script>\n            ')
        assert self.console.log.lines[-2:] == ['hello from A', 'hello from B']

    @pytest.mark.skip('NEXT: emit an error if fetch fails')
    def test_paths_that_do_not_exist(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <py-config>\n                [[fetch]]\n                files = ["./f.py"]\n            </py-config>\n\n            <script type="py">\n                print("this should not be printed")\n            </script>\n            ', wait_for_pyscript=False)
        expected = '(PY0404): Fetching from URL ./f.py failed with error 404'
        inner_html = self.page.locator('.py-error').inner_html()
        assert expected in inner_html
        assert expected in self.console.error.lines[-1]
        assert self.console.log.lines == []

    def test_paths_from_packages(self):
        if False:
            while True:
                i = 10
        self.writefile('utils/__init__.py', '')
        self.writefile('utils/a.py', "x = 'hello from A'")
        self.pyscript_run('\n            <py-config>\n                [[fetch]]\n                from = "utils"\n                to_folder = "pkg"\n                files = ["__init__.py", "a.py"]\n            </py-config>\n\n            <script type="py">\n                import js\n                from pkg.a import x\n                js.console.log(x)\n            </script>\n            ')
        assert self.console.log.lines[-1] == 'hello from A'
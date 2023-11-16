import pytest
from .support import PyScriptTest

@pytest.mark.xfail(reason='See PR #938')
class TestImportmap(PyScriptTest):

    def test_importmap(self):
        if False:
            return 10
        src = '\n            export function say_hello(who) {\n                console.log("hello from", who);\n            }\n        '
        self.writefile('mymod.js', src)
        self.pyscript_run('\n            <script type="importmap">\n            {\n              "imports": {\n                "mymod": "/mymod.js"\n              }\n            }\n            </script>\n\n            <script type="module">\n                import { say_hello } from "mymod";\n                say_hello("JS");\n            </script>\n\n            <script type="py">\n                import mymod\n                mymod.say_hello("Python")\n            </script>\n            ')
        assert self.console.log.lines == ['hello from JS', 'hello from Python']

    def test_invalid_json(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="importmap">\n            this is not valid JSON\n            </script>\n\n            <script type="py">\n                print("hello world")\n            </script>\n            ', wait_for_pyscript=False)
        self.check_js_errors('Failed to parse import map')
        self.wait_for_pyscript()
        assert self.console.log.lines == ['hello world']
        banner = self.page.locator('.py-warning')
        assert 'Failed to parse import map' in banner.inner_text()
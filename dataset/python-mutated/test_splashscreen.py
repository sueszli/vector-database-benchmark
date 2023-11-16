import pytest
from playwright.sync_api import expect
from .support import PyScriptTest, skip_worker
pytest.skip(reason='NEXT: Should we remove the splashscreen?', allow_module_level=True)

class TestSplashscreen(PyScriptTest):

    def test_autoshow_and_autoclose(self):
        if False:
            print('Hello World!')
        '\n        By default, we show the splashscreen and we close it when the loading is\n        complete.\n\n        XXX: this test is a bit fragile: now it works reliably because the\n        startup is so slow that when we do expect(div).to_be_visible(), the\n        splashscreen is still there. But in theory, if the startup become very\n        fast, it could happen that by the time we arrive in python lang, it\n        has already been removed.\n        '
        self.pyscript_run('\n            <script type="py">\n                print(\'hello pyscript\')\n            </script>\n            ', wait_for_pyscript=False)
        div = self.page.locator('py-splashscreen > div')
        expect(div).to_be_visible()
        expect(div).to_contain_text('Python startup...')
        assert 'Python startup...' in self.console.info.text
        self.wait_for_pyscript()
        expect(div).to_be_hidden()
        assert self.page.locator('py-locator').count() == 0
        assert 'hello pyscript' in self.console.log.lines

    def test_autoclose_false(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <py-config>\n                [splashscreen]\n                autoclose = false\n            </py-config>\n            <script type="py">\n                print(\'hello pyscript\')\n            </script>\n            ')
        div = self.page.locator('py-splashscreen > div')
        expect(div).to_be_visible()
        expect(div).to_contain_text('Python startup...')
        expect(div).to_contain_text('Startup complete')
        assert 'hello pyscript' in self.console.log.lines

    def test_autoclose_loader_deprecated(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <py-config>\n                autoclose_loader = false\n            </py-config>\n            <script type="py">\n                print(\'hello pyscript\')\n            </script>\n            ')
        warning = self.page.locator('.py-warning')
        inner_text = warning.inner_html()
        assert 'The setting autoclose_loader is deprecated' in inner_text
        div = self.page.locator('py-splashscreen > div')
        expect(div).to_be_visible()
        expect(div).to_contain_text('Python startup...')
        expect(div).to_contain_text('Startup complete')
        assert 'hello pyscript' in self.console.log.lines

    def test_splashscreen_disabled_option(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <py-config>\n                [splashscreen]\n                enabled = false\n            </py-config>\n\n            <script type="py">\n                def test():\n                    print("Hello pyscript!")\n                test()\n            </script>\n            ')
        assert self.page.locator('py-splashscreen').count() == 0
        assert self.console.log.lines[-1] == 'Hello pyscript!'
        py_terminal = self.page.wait_for_selector('py-terminal')
        assert py_terminal.inner_text() == 'Hello pyscript!\n'

    @skip_worker('FIXME: js.document')
    def test_splashscreen_custom_message(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <py-config>\n                [splashscreen]\n                    autoclose = false\n            </py-config>\n\n            <script type="py">\n                from js import document\n\n                splashscreen = document.querySelector("py-splashscreen")\n                splashscreen.log("Hello, world!")\n            </script>\n            ')
        splashscreen = self.page.locator('py-splashscreen')
        assert splashscreen.count() == 1
        assert 'Hello, world!' in splashscreen.inner_text()
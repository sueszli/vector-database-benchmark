import pytest
from .support import PyScriptTest, skip_worker

class TestWarningsAndBanners(PyScriptTest):

    def test_deprecate_loading_scripts_from_latest(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="py">\n                print("whatever..")\n            </script>\n            ', extra_head='<script type="ignore-me" src="https://pyscript.net/latest/any-path-triggers-the-warning-anyway.js"></script>')
        loc = self.page.wait_for_selector('.py-error')
        assert loc.inner_text() == 'Loading scripts from latest is deprecated and will be removed soon. Please use a specific version instead.'
        loc = self.page.locator('.py-error')
        assert loc.count() == 1

    @pytest.mark.skip('NEXT: To check if behaviour is consistent with classic')
    def test_create_singular_warning(self):
        if False:
            return 10
        self.pyscript_run('\n            <script type="py" output="foo">\n                print("one.")\n                print("two.")\n            </script>\n            <script type="py" output="foo">\n                print("three.")\n            </script>\n            ')
        loc = self.page.locator('.alert-banner')
        assert loc.count() == 1
        assert loc.text_content() == 'output = "foo" does not match the id of any element on the page.'
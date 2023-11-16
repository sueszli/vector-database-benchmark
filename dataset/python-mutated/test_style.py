import pytest
from playwright.sync_api import expect
from .support import PyScriptTest, with_execution_thread

@with_execution_thread(None)
class TestStyle(PyScriptTest):

    def test_pyscript_not_defined(self):
        if False:
            i = 10
            return i + 15
        'Test raw elements that are not defined for display:none'
        doc = '\n        <html>\n          <head>\n              <link rel="stylesheet" href="build/core.css" />\n          </head>\n          <body>\n            <py-config>hello</py-config>\n            <py-script>hello</script>\n          </body>\n        </html>\n        '
        self.writefile('test-not-defined-css.html', doc)
        self.goto('test-not-defined-css.html')
        expect(self.page.locator('py-config')).to_be_hidden()
        expect(self.page.locator('py-script')).to_be_hidden()
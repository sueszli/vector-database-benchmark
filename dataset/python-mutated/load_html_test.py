import pytest
from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

@pytest.mark.offline
class OfflineTests(BaseCase):

    def test_load_html_string(self):
        if False:
            i = 10
            return i + 15
        html = '<h2>Hello</h2><p><input />&nbsp;&nbsp;<button>OK!</button></p>'
        self.load_html_string(html)
        self.assert_text('Hello', 'h2')
        self.assert_text('OK!', 'button')
        self.type('input', 'Goodbye')
        self.click('button')
        new_html = '<h3>Checkbox</h3><p><input type="checkbox" />Check Me!</p>'
        self.set_content(new_html)
        self.assert_text('Checkbox', 'h3')
        self.assert_text('Check Me!', 'p')
        self.assert_false(self.is_selected('input'))
        self.click('input')
        self.assert_true(self.is_selected('input'))
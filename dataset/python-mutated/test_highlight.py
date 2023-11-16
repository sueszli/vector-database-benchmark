from helium import highlight, Button, Text, Config
from helium._impl.util.lang import TemporaryAttrValue
from tests.api import BrowserAT

class HighlightTest(BrowserAT):

    def get_page(self):
        if False:
            return 10
        return 'test_gui_elements.html'

    def test_highlight(self):
        if False:
            return 10
        button = Button('Input Button')
        highlight(button)
        self._check_is_highlighted(button)

    def test_highlight_string(self):
        if False:
            while True:
                i = 10
        highlight('Text with id')
        self._check_is_highlighted(Text('Text with id'))

    def test_highlight_nonexistent(self):
        if False:
            i = 10
            return i + 15
        with TemporaryAttrValue(Config, 'implicit_wait_secs', 0.5):
            with self.assertRaises(LookupError):
                highlight(Button('foo'))

    def _check_is_highlighted(self, html_element):
        if False:
            for i in range(10):
                print('nop')
        style = html_element.web_element.get_attribute('style')
        self.assertTrue('border: 2px solid red;' in style, style)
        self.assertTrue('font-weight: bold;' in style, style)
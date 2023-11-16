from helium import Text, get_driver
from tests.api import BrowserAT

class IframeTest(BrowserAT):

    def get_page(self):
        if False:
            return 10
        return 'test_iframe/main.html'

    def test_test_text_in_iframe_exists(self):
        if False:
            while True:
                i = 10
        self.assertTrue(Text('This text is inside an iframe.').exists())

    def test_text_in_nested_iframe_exists(self):
        if False:
            while True:
                i = 10
        self.assertTrue(Text('This text is inside a nested iframe.').exists())

    def test_finds_element_in_parent_iframe(self):
        if False:
            return 10
        self.test_text_in_nested_iframe_exists()
        self.test_test_text_in_iframe_exists()

    def test_access_attributes_across_iframes(self):
        if False:
            print('Hello World!')
        text = Text('This text is inside an iframe.')
        self.assertEqual('This text is inside an iframe.', text.value)
        get_driver().switch_to.default_content()
        self.assertEqual('This text is inside an iframe.', text.value)
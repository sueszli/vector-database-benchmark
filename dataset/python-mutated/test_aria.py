from helium import Button, TextField
from tests.api import BrowserAT

class AriaTest(BrowserAT):

    def get_page(self):
        if False:
            while True:
                i = 10
        return 'test_aria.html'

    def test_aria_label_button_exists(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(Button('Close').exists())

    def test_aria_label_button_is_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(Button('Close').is_enabled())

    def test_aria_label_disabled_button_is_enabled(self):
        if False:
            return 10
        self.assertFalse(Button('Disabled Close').is_enabled())

    def test_aria_label_non_existent_button(self):
        if False:
            while True:
                i = 10
        self.assertFalse(Button('This doesnt exist').exists())

    def test_aria_label_div_button_exists(self):
        if False:
            return 10
        self.assertTrue(Button('Attach files').exists())

    def test_aria_label_div_button_is_enabled(self):
        if False:
            return 10
        self.assertTrue(Button('Attach files').is_enabled())

    def test_aria_label_div_disabled_button_is_enabled(self):
        if False:
            print('Hello World!')
        self.assertFalse(Button('Disabled Attach files').is_enabled())

    def test_aria_label_submit_button_exists(self):
        if False:
            print('Hello World!')
        self.assertTrue(Button('Submit').exists())

    def test_aria_textbox_exists(self):
        if False:
            print('Hello World!')
        self.assertTrue(TextField('Textbox').exists())

    def test_aria_textbox_value(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('Textbox value', TextField('Textbox').value)
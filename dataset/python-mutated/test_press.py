from helium import press, TextField, SHIFT
from tests.api import BrowserAT

class PressTest(BrowserAT):

    def get_page(self):
        if False:
            i = 10
            return i + 15
        return 'test_write.html'

    def test_press_single_character(self):
        if False:
            i = 10
            return i + 15
        press('a')
        self.assertEqual('a', TextField('Autofocus text field').value)

    def test_press_upper_case_character(self):
        if False:
            print('Hello World!')
        press('A')
        self.assertEqual('A', TextField('Autofocus text field').value)

    def test_press_shift_plus_lower_case_character(self):
        if False:
            while True:
                i = 10
        press(SHIFT + 'a')
        self.assertEqual('A', TextField('Autofocus text field').value)
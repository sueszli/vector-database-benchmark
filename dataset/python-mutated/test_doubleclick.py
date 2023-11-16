from helium import doubleclick
from tests.api import BrowserAT

class DoubleclickTest(BrowserAT):

    def get_page(self):
        if False:
            i = 10
            return i + 15
        return 'test_doubleclick.html'

    def test_double_click(self):
        if False:
            i = 10
            return i + 15
        doubleclick('Doubleclick here.')
        self.assertEqual('Success!', self.read_result_from_browser())
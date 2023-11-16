from helium import click, rightclick
from tests.api import BrowserAT

class RightclickTest(BrowserAT):

    def get_page(self):
        if False:
            for i in range(10):
                print('nop')
        return 'test_rightclick.html'

    def test_simple_rightclick(self):
        if False:
            while True:
                i = 10
        rightclick('Perform a normal rightclick here.')
        self.assertEqual('Normal rightclick performed.', self.read_result_from_browser())

    def test_rightclick_select_normal_item(self):
        if False:
            print('Hello World!')
        rightclick('Rightclick here for context menu.')
        click('Normal item')
        self.assertEqual('Normal item selected.', self.read_result_from_browser())
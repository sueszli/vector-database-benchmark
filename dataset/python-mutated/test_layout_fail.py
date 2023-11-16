"""Visual Layout Testing with different Syntax Formats"""
from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

class VisualLayout_FixtureTests:

    def test_python_home_change(self, sb):
        if False:
            while True:
                i = 10
        sb.open('https://python.org/')
        print('\nCreating baseline in "visual_baseline" folder.')
        sb.check_window(name='python_home', baseline=True)
        sb.remove_element('a.donate-button')
        print('(This test should fail)')
        sb.check_window(name='python_home', level=3)

class VisualLayoutFailureTests(BaseCase):

    def test_applitools_change(self):
        if False:
            return 10
        self.open('https://applitools.com/helloworld/?diff1')
        print('\nCreating baseline in "visual_baseline" folder.')
        self.check_window(name='helloworld', baseline=True)
        self.click('a[href="?diff1"]')
        self.slow_click('button')
        print('(This test should fail)')
        self.check_window(name='helloworld', level=3)

    def test_xkcd_logo_change(self):
        if False:
            while True:
                i = 10
        self.open('https://xkcd.com/554/')
        print('\nCreating baseline in "visual_baseline" folder.')
        self.check_window(name='xkcd_554', baseline=True)
        self.set_attribute('[alt="xkcd.com logo"]', 'height', '110')
        self.set_attribute('[alt="xkcd.com logo"]', 'width', '120')
        print('(This test should fail)')
        self.check_window(name='xkcd_554', level=3)
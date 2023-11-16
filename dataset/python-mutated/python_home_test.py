from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

class VisualLayoutTests(BaseCase):

    def test_python_home_layout_change(self):
        if False:
            for i in range(10):
                print('nop')
        self.open('https://python.org/')
        print('\nCreating baseline in "visual_baseline" folder.')
        self.check_window(name='python_home', baseline=True)
        self.remove_element('a.donate-button')
        self.check_window(name='python_home', level=0)
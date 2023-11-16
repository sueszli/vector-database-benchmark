from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

class VisualLayoutTests(BaseCase):

    def test_xkcd_layout_change(self):
        if False:
            i = 10
            return i + 15
        self.open('https://xkcd.com/554/')
        print('\nCreating baseline in "visual_baseline" folder.')
        self.check_window(name='xkcd_554', baseline=True)
        self.set_attribute('[alt="xkcd.com logo"]', 'height', '130')
        self.set_attribute('[alt="xkcd.com logo"]', 'width', '120')
        self.check_window(name='xkcd_554', level=0)
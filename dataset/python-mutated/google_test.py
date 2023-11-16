"""google.com example test that uses page objects"""
from seleniumbase import BaseCase
try:
    from .google_objects import HomePage, ResultsPage
except Exception:
    from google_objects import HomePage, ResultsPage
    BaseCase.main(__name__, __file__)

class GoogleTests(BaseCase):

    def test_google_dot_com(self):
        if False:
            while True:
                i = 10
        if self.headless and self._multithreaded:
            self.open_if_not_url('about:blank')
            print('Skipping test in headless multi-threaded mode.')
            self.skip('Skipping test in headless multi-threaded mode.')
        self.open('https://google.com/ncr')
        self.assert_title_contains('Google')
        self.sleep(0.05)
        self.save_screenshot_to_logs()
        self.wait_for_element('iframe[role="presentation"]')
        self.hide_elements('iframe')
        self.sleep(0.05)
        self.save_screenshot_to_logs()
        self.type(HomePage.search_box, 'github.com')
        self.assert_element(HomePage.search_button)
        self.assert_element(HomePage.feeling_lucky_button)
        self.click(HomePage.search_button)
        self.assert_text('github.com', ResultsPage.search_results)
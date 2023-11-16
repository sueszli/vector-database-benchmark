from pprint import pformat
from seleniumbase import BaseCase
BaseCase.main(__name__, __file__, '--uc', '--uc-cdp', '-s')

class CDPTests(BaseCase):

    def add_cdp_listener(self):
        if False:
            while True:
                i = 10
        self.driver.add_cdp_listener('Network.requestWillBeSentExtraInfo', lambda data: print(pformat(data)))

    def verify_success(self):
        if False:
            return 10
        self.assert_text('OH YEAH, you passed!', 'h1', timeout=6.25)
        self.sleep(1)

    def fail_me(self):
        if False:
            return 10
        self.fail('Selenium was detected! Try using: "pytest --uc"')

    def test_display_cdp_events(self):
        if False:
            i = 10
            return i + 15
        if not (self.undetectable and self.uc_cdp_events):
            self.get_new_driver(undetectable=True, uc_cdp_events=True)
        self.driver.get('https://nowsecure.nl/#relax')
        try:
            self.verify_success()
        except Exception:
            self.clear_all_cookies()
            self.get_new_driver(undetectable=True, uc_cdp_events=True)
            self.driver.get('https://nowsecure.nl/#relax')
            try:
                self.verify_success()
            except Exception:
                if self.is_element_visible('iframe[src*="challenge"]'):
                    with self.frame_switch('iframe[src*="challenge"]'):
                        self.click('span.mark')
                else:
                    self.fail_me()
                try:
                    self.verify_success()
                except Exception:
                    self.fail_me()
        self.add_cdp_listener()
        self.refresh()
        self.sleep(1)
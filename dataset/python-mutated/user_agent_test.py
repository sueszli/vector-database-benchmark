from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

class UserAgentTests(BaseCase):

    def test_user_agent(self):
        if False:
            i = 10
            return i + 15
        if self._multithreaded:
            self.open_if_not_url('about:blank')
            self.skip('Skipping test in multi-threaded mode.')
        self.open('https://my-user-agent.com/')
        zoom_in = '#ua_string{zoom: 1.8;-moz-transform: scale(1.8);}'
        self.add_css_style(zoom_in)
        self.highlight('#ua_string')
        user_agent_detected = self.get_text('#ua_string')
        original_user_agent = user_agent_detected
        if not self.user_agent:
            print('\n\nUser-Agent: %s' % user_agent_detected)
        else:
            print('\n\nUser-Agent override: %s' % user_agent_detected)
        if self.headed:
            self.sleep(2.75)
        if not self.is_chromium():
            msg = '\n* execute_cdp_cmd() is only for Chromium browsers'
            print(msg)
            self.skip(msg)
        print('\n--------------------------')
        try:
            self.execute_cdp_cmd('Network.setUserAgentOverride', {'userAgent': 'Mozilla/5.0 (Nintendo Switch; WifiWebAuthApplet) AppleWebKit/606.4 (KHTML, like Gecko) NF/6.0.1.15.4 NintendoBrowser/5.1.0.20393'})
            self.open('about:blank')
            self.sleep(0.05)
            self.open('https://my-user-agent.com/')
            zoom_in = '#ua_string{zoom: 1.8;-moz-transform: scale(1.8);}'
            self.add_css_style(zoom_in)
            self.highlight('#ua_string')
            user_agent_detected = self.get_text('#ua_string')
            print('\nUser-Agent override: %s' % user_agent_detected)
            if self.headed:
                self.sleep(2.75)
        finally:
            self.execute_cdp_cmd('Network.setUserAgentOverride', {'userAgent': original_user_agent})
from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

class TestMFALogin(BaseCase):

    def test_mfa_login(self):
        if False:
            while True:
                i = 10
        self.open('https://seleniumbase.io/realworld/login')
        self.type('#username', 'demo_user')
        self.type('#password', 'secret_pass')
        self.enter_mfa_code('#totpcode', 'GAXG2MTEOR3DMMDG')
        self.assert_text('Welcome!', 'h1')
        self.highlight('img#image1')
        self.click('a:contains("This Page")')
        self.save_screenshot_to_logs()
        self.click_link('Sign out')
        self.assert_element('a:contains("Sign in")')
        self.assert_exact_text('You have been signed out!', '#top_message')
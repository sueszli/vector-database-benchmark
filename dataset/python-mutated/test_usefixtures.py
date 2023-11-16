import pytest

@pytest.mark.usefixtures('sb')
class Test_UseFixtures:

    def test_usefixtures_on_class(self):
        if False:
            return 10
        sb = self.sb
        sb.open('https://seleniumbase.io/realworld/login')
        sb.type('#username', 'demo_user')
        sb.type('#password', 'secret_pass')
        sb.enter_mfa_code('#totpcode', 'GAXG2MTEOR3DMMDG')
        sb.assert_text('Welcome!', 'h1')
        sb.highlight('img#image1')
        sb.click('a:contains("This Page")')
        sb.save_screenshot_to_logs()
        sb.click_link('Sign out')
        sb.assert_element('a:contains("Sign in")')
        sb.assert_exact_text('You have been signed out!', '#top_message')
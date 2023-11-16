from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

class HandleAlertTests(BaseCase):

    def test_alerts(self):
        if False:
            for i in range(10):
                print('nop')
        self.open('about:blank')
        self.execute_script('window.alert("ALERT!!!");')
        self.sleep(1)
        self.accept_alert()
        self.sleep(1)
        self.execute_script('window.prompt("My Prompt","defaultText");')
        self.sleep(1)
        alert = self.switch_to_alert()
        self.assert_equal(alert.text, 'My Prompt')
        self.dismiss_alert()
        self.sleep(1)
        if self.browser == 'safari' and self._reuse_session:
            self.driver.quit()
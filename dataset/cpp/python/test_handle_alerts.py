import pytest
from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)


@pytest.mark.offline  # Can be run with: "pytest -m offline"
class OfflineTests(BaseCase):
    def test_alerts(self):
        self.open("data:,")
        self.execute_script('window.alert("ALERT!!!");')
        self.sleep(1)  # Not needed (Lets you see the alert pop up)
        self.accept_alert()
        self.sleep(1)  # Not needed (Lets you see the alert go away)
        self.execute_script('window.prompt("My Prompt","defaultText");')
        self.sleep(1)  # Not needed (Lets you see the alert pop up)
        alert = self.switch_to_alert()
        self.assert_equal(alert.text, "My Prompt")  # Not input field
        self.dismiss_alert()
        self.sleep(1)  # Not needed (Lets you see the alert go away)
        if self.browser == "safari" and self._reuse_session:
            # Alerts can freeze Safari if reusing the browser session
            self.driver.quit()

from .base import BasePage

class TransactionSummaryPage(BasePage):

    def wait_until_loaded(self):
        if False:
            while True:
                i = 10
        self.browser.wait_until_not('[data-test-id="loading-indicator"]')
        self.browser.wait_until_not('[data-test-id="loading-placeholder"]')
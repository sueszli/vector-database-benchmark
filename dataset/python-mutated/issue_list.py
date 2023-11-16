from .base import BasePage
from .global_selection import GlobalSelectionPage
from .issue_details import IssueDetailsPage

class IssueListPage(BasePage):

    def __init__(self, browser, client):
        if False:
            return 10
        super().__init__(browser)
        self.client = client
        self.global_selection = GlobalSelectionPage(browser)

    def visit_issue_list(self, org, query=''):
        if False:
            print('Hello World!')
        self.browser.get(f'/organizations/{org}/issues/{query}')
        self.wait_until_loaded()

    def wait_for_stream(self):
        if False:
            print('Hello World!')
        self.browser.wait_until('[data-test-id="event-issue-header"]', timeout=20)

    def select_issue(self, position):
        if False:
            return 10
        self.browser.click(f'[data-test-id="group"]:nth-child({position})')

    def navigate_to_issue(self, position):
        if False:
            while True:
                i = 10
        self.browser.click(f'[data-test-id="group"]:nth-child({position}) a')
        self.browser.wait_until('.group-detail')
        self.issue_details = IssueDetailsPage(self.browser, self.client)

    def resolve_issues(self):
        if False:
            return 10
        self.browser.click('[aria-label="Resolve"]')

    def wait_for_issue_removal(self):
        if False:
            i = 10
            return i + 15
        self.browser.click_when_visible('[data-test-id="toast-success"]')
        self.browser.wait_until_not('[data-test-id="toast-success"]')

    def wait_for_issue(self):
        if False:
            print('Hello World!')
        self.browser.wait_until('[data-test-id="group"]')

    def find_resolved_issues(self):
        if False:
            print('Hello World!')
        return self.browser.elements('[data-test-id="resolved-issue"]')

    def ignore_issues(self):
        if False:
            for i in range(10):
                print('nop')
        self.browser.click('[aria-label="Ignore"]')

    def delete_issues(self):
        if False:
            i = 10
            return i + 15
        self.browser.click('[aria-label="More issue actions"]')
        self.browser.wait_until('[data-test-id="delete"]')
        self.browser.click('[data-test-id="delete"]')
        self.browser.click('[data-test-id="confirm-button"]')

    def merge_issues(self):
        if False:
            return 10
        self.browser.click('[aria-label="Merge Selected Issues"]')
        self.browser.click('[data-test-id="confirm-button"]')

    def mark_reviewed_issues(self):
        if False:
            while True:
                i = 10
        self.browser.click('[aria-label="Mark Reviewed"]')
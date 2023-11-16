from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from fixtures.page_objects.issue_list import IssueListPage
from sentry.models.savedsearch import SavedSearch, SortOptions, Visibility
from sentry.testutils.cases import AcceptanceTestCase, SnubaTestCase
from sentry.testutils.silo import no_silo_test

@no_silo_test(stable=True)
class OrganizationGroupIndexTest(AcceptanceTestCase, SnubaTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.user = self.create_user('foo@example.com')
        self.org = self.create_organization(owner=self.user, name='Rowdy Tiger')
        self.team = self.create_team(organization=self.org, name='Mariachi Band', members=[self.user])
        self.project = self.create_project(organization=self.org, teams=[self.team], name='Bengal')
        self.other_project = self.create_project(organization=self.org, teams=[self.team], name='Sumatra')
        self.login_as(self.user)
        self.page = IssueListPage(self.browser, self.client)
        self.dismiss_assistant()
        self.create_saved_search(name='Assigned to Me', query='is:unresolved assigned:me', visibility=Visibility.ORGANIZATION, is_global=True)
        self.create_saved_search(name='Errors Only', query='is:unresolved evel:error', visibility=Visibility.ORGANIZATION, is_global=True)

    def test_click_saved_search(self):
        if False:
            for i in range(10):
                print('nop')
        self.page.visit_issue_list(self.org.slug)
        self.browser.click_when_visible('button[aria-label="Custom Search"]')
        self.browser.click('button[aria-label="Errors Only"]')
        self.page.wait_until_loaded()

    def test_create_saved_search(self):
        if False:
            i = 10
            return i + 15
        self.page.visit_issue_list(self.org.slug)
        self.browser.click_when_visible('button[aria-label="Custom Search"]')
        self.browser.click('[aria-label="Add saved search"]')
        self.browser.wait_until('[role="dialog"]')
        self.browser.find_element(by=By.NAME, value='name').send_keys('My Saved Search')
        query_input = self.browser.find_element(by=By.CSS_SELECTOR, value='[role="dialog"] textarea')
        self.browser.click('[role="dialog"] button[aria-label="Clear search"]')
        query_input.send_keys('browser.name:Firefox', Keys.ENTER)
        self.browser.click('[role="dialog"] button[aria-label="Save"]')
        self.browser.wait_until_not('[data-test-id="loading-indicator"]')
        created_search = SavedSearch.objects.get(name='My Saved Search')
        assert created_search
        assert created_search.query == 'browser.name:Firefox'
        assert created_search.sort == SortOptions.DATE
        assert created_search.visibility == Visibility.OWNER
        assert not created_search.is_global
        assert created_search.owner_id == self.user.id
        assert self.browser.find_element(by=By.CSS_SELECTOR, value='button[aria-label="My Saved Search"]')

    def test_edit_saved_search(self):
        if False:
            print('Hello World!')
        self.create_saved_search(organization=self.org, name='My Saved Search', query='browser.name:Firefox', visibility=Visibility.OWNER, owner=self.user)
        self.page.visit_issue_list(self.org.slug)
        self.browser.click_when_visible('button[aria-label="Custom Search"]')
        self.browser.move_to('button[aria-label="My Saved Search"]')
        self.browser.wait_until_clickable('button[aria-label="Saved search options"]')
        self.browser.click('button[aria-label="Saved search options"]')
        self.browser.click('[data-test-id="edit"]')
        self.browser.wait_until('[role="dialog"]')
        self.browser.find_element(by=By.NAME, value='name').clear()
        self.browser.find_element(by=By.NAME, value='name').send_keys('New Saved Search Name')
        self.browser.click('[role="dialog"] button[aria-label="Save"]')
        self.browser.wait_until_not('[data-test-id="loading-indicator"]')
        created_search = SavedSearch.objects.get(name='New Saved Search Name')
        assert created_search
        assert created_search.query == 'browser.name:Firefox'
        assert created_search.sort == SortOptions.DATE
        assert created_search.visibility == Visibility.OWNER
        assert not created_search.is_global
        assert created_search.owner_id == self.user.id
        assert self.browser.find_element(by=By.CSS_SELECTOR, value='button[aria-label="New Saved Search Name"]')

    def test_delete_saved_search(self):
        if False:
            for i in range(10):
                print('nop')
        self.create_saved_search(organization=self.org, name='My Saved Search', query='browser.name:Firefox', visibility=Visibility.OWNER, owner=self.user)
        self.page.visit_issue_list(self.org.slug)
        self.browser.click_when_visible('button[aria-label="Custom Search"]')
        self.browser.move_to('button[aria-label="My Saved Search"]')
        self.browser.wait_until_clickable('button[aria-label="Saved search options"]')
        self.browser.click('button[aria-label="Saved search options"]')
        self.browser.click('[data-test-id="delete"]')
        self.browser.wait_until('[role="dialog"]')
        self.browser.click('[role="dialog"] button[aria-label="Confirm"]')
        assert not self.browser.element_exists('button[aria-label="My Saved Search"]')
        wait = WebDriverWait(self.browser.driver, 10)
        wait.until(lambda _: not SavedSearch.objects.filter(name='My Saved Search').exists())
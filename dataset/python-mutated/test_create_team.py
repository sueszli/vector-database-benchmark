from selenium.webdriver.common.by import By
from sentry.models.team import Team
from sentry.testutils.cases import AcceptanceTestCase
from sentry.testutils.silo import no_silo_test

@no_silo_test(stable=True)
class CreateTeamTest(AcceptanceTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.user = self.create_user('foo@example.com')
        self.org = self.create_organization(name='Rowdy Tiger', owner=None)
        self.team = self.create_team(organization=self.org, name='Mariachi Band')
        self.project = self.create_project(organization=self.org, teams=[self.team], name='Bengal')
        self.create_member(user=self.user, organization=self.org, role='owner', teams=[self.team])
        self.login_as(self.user)
        self.path = f'/settings/{self.org.slug}/teams/'

    def test_create(self):
        if False:
            while True:
                i = 10
        self.browser.get(self.path)
        self.browser.wait_until_test_id('team-list')
        self.browser.click('button[aria-label="Create Team"]')
        self.browser.wait_until("[role='dialog']")
        self.browser.element('input[id="slug"]').send_keys('new-team')
        self.browser.click("[role='dialog'] button[aria-label='Create Team']")
        self.browser.wait_until_not("[role='dialog']")
        assert self.browser.find_element(by=By.XPATH, value="//span[contains(text(), 'new-team')]")
        assert Team.objects.filter(slug='new-team', organization=self.org).exists()
from datetime import timedelta
from django.utils import timezone
from sentry.models.rule import Rule
from sentry.models.rulefirehistory import RuleFireHistory
from sentry.testutils.cases import AcceptanceTestCase, SnubaTestCase
from sentry.testutils.silo import no_silo_test

@no_silo_test(stable=True)
class OrganizationAlertRuleDetailsTest(AcceptanceTestCase, SnubaTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.login_as(self.user)
        self.rule = Rule.objects.filter(project=self.project).first()
        self.path = f'/organizations/{self.organization.slug}/alerts/rules/{self.project.slug}/{self.rule.id}/details/'

    def test_empty_alert_rule_details(self):
        if False:
            return 10
        self.browser.get(self.path)
        self.browser.wait_until_not('[data-test-id="loading-indicator"]')

    def test_alert_rule_with_issues(self):
        if False:
            return 10
        group = self.create_group()
        RuleFireHistory.objects.create(project=self.project, rule=self.rule, group=group, date_added=timezone.now() - timedelta(days=1))
        self.browser.get(self.path)
        self.browser.wait_until_not('[data-test-id="loading-indicator"]')
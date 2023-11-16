from sentry.rules.filters.issue_occurrences import IssueOccurrencesFilter
from sentry.testutils.cases import RuleTestCase
from sentry.testutils.silo import region_silo_test
from sentry.testutils.skips import requires_snuba
pytestmark = [requires_snuba]

@region_silo_test(stable=True)
class IssueOccurrencesTest(RuleTestCase):
    rule_cls = IssueOccurrencesFilter

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.event.group.times_seen_pending = 0

    def test_compares_correctly(self):
        if False:
            i = 10
            return i + 15
        event = self.get_event()
        value = 10
        data = {'value': str(value)}
        rule = self.get_rule(data=data)
        event.group.times_seen = 11
        self.assertPasses(rule, event)
        event.group.times_seen = 10
        self.assertPasses(rule, event)
        event.group.times_seen = 8
        self.assertDoesNotPass(rule, event)

    def test_uses_pending(self):
        if False:
            print('Hello World!')
        event = self.get_event()
        value = 10
        data = {'value': str(value)}
        rule = self.get_rule(data=data)
        event.group.times_seen = 8
        self.assertDoesNotPass(rule, event)
        event.group.times_seen_pending = 3
        self.assertPasses(rule, event)

    def test_fails_on_bad_data(self):
        if False:
            return 10
        event = self.get_event()
        data = {'value': 'bad data'}
        rule = self.get_rule(data=data)
        event.group.times_seen = 10
        self.assertDoesNotPass(rule, event)
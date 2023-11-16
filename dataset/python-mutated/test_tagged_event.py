from sentry.rules.conditions.tagged_event import TaggedEventCondition
from sentry.rules.match import MatchType
from sentry.testutils.cases import RuleTestCase
from sentry.testutils.silo import region_silo_test
from sentry.testutils.skips import requires_snuba
pytestmark = [requires_snuba]

@region_silo_test(stable=True)
class TaggedEventConditionTest(RuleTestCase):
    rule_cls = TaggedEventCondition

    def get_event(self):
        if False:
            for i in range(10):
                print('nop')
        event = self.event
        event.data['tags'] = (('logger', 'sentry.example'), ('logger', 'foo.bar'), ('notlogger', 'sentry.other.example'), ('notlogger', 'bar.foo.baz'))
        return event

    def test_render_label(self):
        if False:
            i = 10
            return i + 15
        rule = self.get_rule(data={'match': MatchType.EQUAL, 'key': 'Ã', 'value': 'Ä'})
        assert rule.render_label() == "The event's tags match Ã equals Ä"

    def test_equals(self):
        if False:
            return 10
        event = self.get_event()
        rule = self.get_rule(data={'match': MatchType.EQUAL, 'key': 'LOGGER', 'value': 'sentry.example'})
        self.assertPasses(rule, event)
        rule = self.get_rule(data={'match': MatchType.EQUAL, 'key': 'logger', 'value': 'sentry.other.example'})
        self.assertDoesNotPass(rule, event)

    def test_does_not_equal(self):
        if False:
            for i in range(10):
                print('nop')
        event = self.get_event()
        rule = self.get_rule(data={'match': MatchType.NOT_EQUAL, 'key': 'logger', 'value': 'sentry.example'})
        self.assertDoesNotPass(rule, event)
        rule = self.get_rule(data={'match': MatchType.NOT_EQUAL, 'key': 'logger', 'value': 'sentry.other.example'})
        self.assertPasses(rule, event)

    def test_starts_with(self):
        if False:
            i = 10
            return i + 15
        event = self.get_event()
        rule = self.get_rule(data={'match': MatchType.STARTS_WITH, 'key': 'logger', 'value': 'sentry.'})
        self.assertPasses(rule, event)
        rule = self.get_rule(data={'match': MatchType.STARTS_WITH, 'key': 'logger', 'value': 'bar.'})
        self.assertDoesNotPass(rule, event)

    def test_does_not_start_with(self):
        if False:
            print('Hello World!')
        event = self.get_event()
        rule = self.get_rule(data={'match': MatchType.NOT_STARTS_WITH, 'key': 'logger', 'value': 'sentry.'})
        self.assertDoesNotPass(rule, event)
        rule = self.get_rule(data={'match': MatchType.NOT_STARTS_WITH, 'key': 'logger', 'value': 'bar.'})
        self.assertPasses(rule, event)

    def test_ends_with(self):
        if False:
            print('Hello World!')
        event = self.get_event()
        rule = self.get_rule(data={'match': MatchType.ENDS_WITH, 'key': 'logger', 'value': '.example'})
        self.assertPasses(rule, event)
        rule = self.get_rule(data={'match': MatchType.ENDS_WITH, 'key': 'logger', 'value': '.foo'})
        self.assertDoesNotPass(rule, event)

    def test_does_not_end_with(self):
        if False:
            i = 10
            return i + 15
        event = self.get_event()
        rule = self.get_rule(data={'match': MatchType.NOT_ENDS_WITH, 'key': 'logger', 'value': '.example'})
        self.assertDoesNotPass(rule, event)
        rule = self.get_rule(data={'match': MatchType.NOT_ENDS_WITH, 'key': 'logger', 'value': '.foo'})
        self.assertPasses(rule, event)

    def test_contains(self):
        if False:
            i = 10
            return i + 15
        event = self.get_event()
        rule = self.get_rule(data={'match': MatchType.CONTAINS, 'key': 'logger', 'value': 'sentry'})
        self.assertPasses(rule, event)
        rule = self.get_rule(data={'match': MatchType.CONTAINS, 'key': 'logger', 'value': 'bar.foo'})
        self.assertDoesNotPass(rule, event)

    def test_does_not_contain(self):
        if False:
            for i in range(10):
                print('nop')
        event = self.get_event()
        rule = self.get_rule(data={'match': MatchType.NOT_CONTAINS, 'key': 'logger', 'value': 'sentry'})
        self.assertDoesNotPass(rule, event)
        rule = self.get_rule(data={'match': MatchType.NOT_CONTAINS, 'key': 'logger', 'value': 'bar.foo'})
        self.assertPasses(rule, event)

    def test_is_set(self):
        if False:
            print('Hello World!')
        event = self.get_event()
        rule = self.get_rule(data={'match': MatchType.IS_SET, 'key': 'logger'})
        self.assertPasses(rule, event)
        rule = self.get_rule(data={'match': MatchType.IS_SET, 'key': 'missing'})
        self.assertDoesNotPass(rule, event)

    def test_is_not_set(self):
        if False:
            for i in range(10):
                print('nop')
        event = self.get_event()
        rule = self.get_rule(data={'match': MatchType.NOT_SET, 'key': 'logger'})
        self.assertDoesNotPass(rule, event)
        rule = self.get_rule(data={'match': MatchType.NOT_SET, 'key': 'missing'})
        self.assertPasses(rule, event)
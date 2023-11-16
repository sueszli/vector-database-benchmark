from sentry.rules.conditions.level import LevelCondition
from sentry.rules.match import MatchType
from sentry.testutils.cases import RuleTestCase
from sentry.testutils.skips import requires_snuba
pytestmark = [requires_snuba]

class LevelConditionTest(RuleTestCase):
    rule_cls = LevelCondition

    def test_render_label(self):
        if False:
            for i in range(10):
                print('nop')
        rule = self.get_rule(data={'match': MatchType.EQUAL, 'level': '30'})
        assert rule.render_label() == "The event's level is equal to warning"

    def test_equals(self):
        if False:
            i = 10
            return i + 15
        event = self.store_event(data={'level': 'info'}, project_id=self.project.id)
        rule = self.get_rule(data={'match': MatchType.EQUAL, 'level': '20'})
        self.assertPasses(rule, event)
        rule = self.get_rule(data={'match': MatchType.EQUAL, 'level': '30'})
        self.assertDoesNotPass(rule, event)

    def test_greater_than(self):
        if False:
            while True:
                i = 10
        event = self.store_event(data={'level': 'info'}, project_id=self.project.id)
        rule = self.get_rule(data={'match': MatchType.GREATER_OR_EQUAL, 'level': '40'})
        self.assertDoesNotPass(rule, event)
        rule = self.get_rule(data={'match': MatchType.GREATER_OR_EQUAL, 'level': '20'})
        self.assertPasses(rule, event)

    def test_less_than(self):
        if False:
            print('Hello World!')
        event = self.store_event(data={'level': 'info'}, project_id=self.project.id)
        rule = self.get_rule(data={'match': MatchType.LESS_OR_EQUAL, 'level': '10'})
        self.assertDoesNotPass(rule, event)
        rule = self.get_rule(data={'match': MatchType.LESS_OR_EQUAL, 'level': '30'})
        self.assertPasses(rule, event)

    def test_without_tag(self):
        if False:
            print('Hello World!')
        event = self.store_event(data={}, project_id=self.project.id)
        rule = self.get_rule(data={'match': MatchType.EQUAL, 'level': '30'})
        self.assertDoesNotPass(rule, event)

    def test_differing_levels(self):
        if False:
            return 10
        eevent = self.store_event(data={'level': 'error'}, project_id=self.project.id)
        wevent = self.store_event(data={'level': 'warning'}, project_id=self.project.id)
        assert wevent.event_id != eevent.event_id
        assert eevent.group is not None
        assert wevent.group is not None
        assert wevent.group.id == eevent.group.id
        rule = self.get_rule(data={'match': MatchType.GREATER_OR_EQUAL, 'level': '40'})
        self.assertDoesNotPass(rule, wevent)
        self.assertPasses(rule, eevent)
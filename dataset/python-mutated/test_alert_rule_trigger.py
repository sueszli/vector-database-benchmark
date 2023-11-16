from sentry.api.serializers import serialize
from sentry.api.serializers.models.alert_rule_trigger import DetailedAlertRuleTriggerSerializer
from sentry.incidents.logic import create_alert_rule_trigger
from sentry.incidents.models import AlertRuleThresholdType
from sentry.testutils.cases import TestCase
NOT_SET = object()

class BaseAlertRuleTriggerSerializerTest:

    def assert_alert_rule_trigger_serialized(self, trigger, result, alert_threshold=NOT_SET):
        if False:
            return 10
        assert result['id'] == str(trigger.id)
        assert result['alertRuleId'] == str(trigger.alert_rule_id)
        assert result['label'] == trigger.label
        assert result['thresholdType'] == trigger.alert_rule.threshold_type
        assert result['alertThreshold'] == (trigger.alert_threshold if alert_threshold is NOT_SET else alert_threshold)
        assert result['resolveThreshold'] == trigger.alert_rule.resolve_threshold
        assert result['dateCreated'] == trigger.date_added

class AlertRuleTriggerSerializerTest(BaseAlertRuleTriggerSerializerTest, TestCase):

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        alert_rule = self.create_alert_rule(resolve_threshold=200)
        trigger = create_alert_rule_trigger(alert_rule, 'hi', 1000)
        result = serialize(trigger)
        self.assert_alert_rule_trigger_serialized(trigger, result)

    def test_decimal(self):
        if False:
            for i in range(10):
                print('nop')
        alert_rule = self.create_alert_rule(resolve_threshold=200.7)
        trigger = create_alert_rule_trigger(alert_rule, 'hi', 1000.5)
        result = serialize(trigger)
        self.assert_alert_rule_trigger_serialized(trigger, result)

    def test_comparison_above(self):
        if False:
            return 10
        alert_rule = self.create_alert_rule(comparison_delta=60)
        trigger = create_alert_rule_trigger(alert_rule, 'hi', 180)
        result = serialize(trigger)
        self.assert_alert_rule_trigger_serialized(trigger, result, 80)

    def test_comparison_below(self):
        if False:
            print('Hello World!')
        alert_rule = self.create_alert_rule(comparison_delta=60, threshold_type=AlertRuleThresholdType.BELOW)
        trigger = create_alert_rule_trigger(alert_rule, 'hi', 80)
        result = serialize(trigger)
        self.assert_alert_rule_trigger_serialized(trigger, result, 20)

class DetailedAlertRuleTriggerSerializerTest(BaseAlertRuleTriggerSerializerTest, TestCase):

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        alert_rule = self.create_alert_rule(resolve_threshold=200)
        trigger = create_alert_rule_trigger(alert_rule, 'hi', 1000)
        result = serialize(trigger, serializer=DetailedAlertRuleTriggerSerializer())
        self.assert_alert_rule_trigger_serialized(trigger, result)
        assert result['excludedProjects'] == []

    def test_excluded_projects(self):
        if False:
            for i in range(10):
                print('nop')
        excluded = [self.create_project()]
        alert_rule = self.create_alert_rule(projects=excluded, resolve_threshold=200)
        trigger = create_alert_rule_trigger(alert_rule, 'hi', 1000, excluded_projects=excluded)
        result = serialize(trigger, serializer=DetailedAlertRuleTriggerSerializer())
        self.assert_alert_rule_trigger_serialized(trigger, result)
        assert result['excludedProjects'] == [p.slug for p in excluded]
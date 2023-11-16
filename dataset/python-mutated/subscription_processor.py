from __future__ import annotations
import logging
import operator
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar, cast
from django.conf import settings
from django.db import router, transaction
from django.utils import timezone
from snuba_sdk import Column, Condition, Limit, Op
from sentry import features
from sentry.constants import CRASH_RATE_ALERT_AGGREGATE_ALIAS, CRASH_RATE_ALERT_SESSION_COUNT_ALIAS
from sentry.incidents.logic import CRITICAL_TRIGGER_LABEL, WARNING_TRIGGER_LABEL, create_incident, deduplicate_trigger_actions, update_incident_status
from sentry.incidents.models import AlertRule, AlertRuleThresholdType, AlertRuleTrigger, Incident, IncidentActivity, IncidentStatus, IncidentStatusMethod, IncidentTrigger, IncidentType, TriggerStatus
from sentry.incidents.tasks import handle_trigger_action
from sentry.incidents.utils.types import SubscriptionUpdate
from sentry.models.project import Project
from sentry.snuba.dataset import Dataset
from sentry.snuba.entity_subscription import ENTITY_TIME_COLUMNS, BaseCrashRateMetricsEntitySubscription, get_entity_key_from_query_builder, get_entity_subscription_from_snuba_query
from sentry.snuba.models import QuerySubscription
from sentry.snuba.tasks import build_query_builder
from sentry.utils import metrics, redis
from sentry.utils.dates import to_datetime, to_timestamp
from sentry.utils.redis import RetryingRedisCluster
logger = logging.getLogger(__name__)
REDIS_TTL = int(timedelta(days=7).total_seconds())
ALERT_RULE_BASE_KEY = '{alert_rule:%s:project:%s}'
ALERT_RULE_BASE_STAT_KEY = '%s:%s'
ALERT_RULE_STAT_KEYS = ('last_update',)
ALERT_RULE_BASE_TRIGGER_STAT_KEY = '%s:trigger:%s:%s'
ALERT_RULE_TRIGGER_STAT_KEYS = ('alert_triggered', 'resolve_triggered')
CRASH_RATE_ALERT_MINIMUM_THRESHOLD: Optional[int] = None
T = TypeVar('T')

class SubscriptionProcessor:
    """
    Class for processing subscription updates for an alert rule. Accepts a subscription
    and then can process one or more updates via `process_update`. Keeps track of how
    close an alert rule is to alerting, creates an incident, and auto resolves the
    incident if a resolve threshold is set and the threshold is triggered.
    """
    THRESHOLD_TYPE_OPERATORS = {AlertRuleThresholdType.ABOVE: (operator.gt, operator.lt), AlertRuleThresholdType.BELOW: (operator.lt, operator.gt)}

    def __init__(self, subscription: QuerySubscription) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.subscription = subscription
        try:
            self.alert_rule = AlertRule.objects.get_for_subscription(subscription)
        except AlertRule.DoesNotExist:
            return
        self.triggers = AlertRuleTrigger.objects.get_for_alert_rule(self.alert_rule)
        self.triggers.sort(key=lambda trigger: trigger.alert_threshold)
        (self.last_update, self.trigger_alert_counts, self.trigger_resolve_counts) = get_alert_rule_stats(self.alert_rule, self.subscription, self.triggers)
        self.orig_trigger_alert_counts = deepcopy(self.trigger_alert_counts)
        self.orig_trigger_resolve_counts = deepcopy(self.trigger_resolve_counts)

    @property
    def active_incident(self) -> Incident:
        if False:
            i = 10
            return i + 15
        if not hasattr(self, '_active_incident'):
            self._active_incident = Incident.objects.get_active_incident(self.alert_rule, self.subscription.project)
        return self._active_incident

    @active_incident.setter
    def active_incident(self, active_incident: Incident) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._active_incident = active_incident

    @property
    def incident_triggers(self) -> Dict[int, IncidentTrigger]:
        if False:
            while True:
                i = 10
        if not hasattr(self, '_incident_triggers'):
            incident = self.active_incident
            incident_triggers = {}
            if incident:
                triggers = IncidentTrigger.objects.filter(incident=incident).select_related('alert_rule_trigger')
                incident_triggers = {trigger.alert_rule_trigger_id: trigger for trigger in triggers}
            self._incident_triggers = incident_triggers
        return self._incident_triggers

    def check_trigger_status(self, trigger: AlertRuleTrigger, status: TriggerStatus) -> bool:
        if False:
            while True:
                i = 10
        '\n        Determines whether a trigger is currently at the specified status\n        :param trigger: An `AlertRuleTrigger`\n        :param status: A `TriggerStatus`\n        :return: True if at the specified status, otherwise False\n        '
        incident_trigger = self.incident_triggers.get(trigger.id)
        return incident_trigger is not None and incident_trigger.status == status.value

    def reset_trigger_counts(self) -> None:
        if False:
            return 10
        '\n        Helper method that clears both the trigger alert and the trigger resolve counts\n        '
        for trigger_id in self.trigger_alert_counts:
            self.trigger_alert_counts[trigger_id] = 0
        for trigger_id in self.trigger_resolve_counts:
            self.trigger_resolve_counts[trigger_id] = 0
        self.update_alert_rule_stats()

    def calculate_resolve_threshold(self, trigger: IncidentTrigger) -> float:
        if False:
            i = 10
            return i + 15
        '\n        Determine the resolve threshold for a trigger. First checks whether an\n        explicit resolve threshold has been set on the rule, and whether this trigger is\n        the lowest severity on the rule. If not, calculates a threshold based on the\n        `alert_threshold` on the trigger.\n        :return:\n        '
        if self.alert_rule.resolve_threshold is not None and (len(self.triggers) == 1 or trigger.label == WARNING_TRIGGER_LABEL):
            resolve_threshold: float = self.alert_rule.resolve_threshold
            return resolve_threshold
        if self.alert_rule.threshold_type == AlertRuleThresholdType.ABOVE.value:
            resolve_add = 1e-06
        else:
            resolve_add = -1e-06
        threshold: float = trigger.alert_threshold + resolve_add
        return threshold

    def get_comparison_aggregation_value(self, subscription_update: SubscriptionUpdate, aggregation_value: float) -> Optional[float]:
        if False:
            print('Hello World!')
        delta = timedelta(seconds=self.alert_rule.comparison_delta)
        end = subscription_update['timestamp'] - delta
        snuba_query = self.subscription.snuba_query
        start = end - timedelta(seconds=snuba_query.time_window)
        entity_subscription = get_entity_subscription_from_snuba_query(snuba_query, self.subscription.project.organization_id)
        try:
            project_ids = [self.subscription.project_id]
            query_builder = build_query_builder(entity_subscription, snuba_query.query, project_ids, snuba_query.environment, params={'organization_id': self.subscription.project.organization.id, 'project_id': project_ids, 'start': start, 'end': end})
            time_col = ENTITY_TIME_COLUMNS[get_entity_key_from_query_builder(query_builder)]
            query_builder.add_conditions([Condition(Column(time_col), Op.GTE, start), Condition(Column(time_col), Op.LT, end)])
            query_builder.limit = Limit(1)
            results = query_builder.run_query(referrer='subscription_processor.comparison_query')
            comparison_aggregate = list(results['data'][0].values())[0]
        except Exception:
            logger.exception('Failed to run comparison query')
            return None
        if not comparison_aggregate:
            metrics.incr('incidents.alert_rules.skipping_update_comparison_value_invalid')
            return None
        result: float = aggregation_value / comparison_aggregate * 100
        return result

    def get_crash_rate_alert_aggregation_value(self, subscription_update: SubscriptionUpdate) -> Optional[float]:
        if False:
            i = 10
            return i + 15
        "\n        Handles validation and extraction of Crash Rate Alerts subscription updates values.\n        The subscription update looks like\n        {\n            '_crash_rate_alert_aggregate': 0.5,\n            '_total_count': 34\n        }\n        - `_crash_rate_alert_aggregate` represents sessions_crashed/sessions or\n        users_crashed/users, and so we need to subtract that number from 1 and then multiply by\n        100 to get the crash free percentage\n        - `_total_count` represents the total sessions or user counts. This is used when\n        CRASH_RATE_ALERT_MINIMUM_THRESHOLD is set in the sense that if the minimum threshold is\n        greater than the session count, then the update is dropped. If the minimum threshold is\n        not set then the total sessions count is just ignored\n        "
        aggregation_value = subscription_update['values']['data'][0][CRASH_RATE_ALERT_AGGREGATE_ALIAS]
        if aggregation_value is None:
            self.reset_trigger_counts()
            metrics.incr('incidents.alert_rules.ignore_update_no_session_data')
            return None
        try:
            total_count = subscription_update['values']['data'][0][CRASH_RATE_ALERT_SESSION_COUNT_ALIAS]
            if CRASH_RATE_ALERT_MINIMUM_THRESHOLD is not None:
                min_threshold = int(CRASH_RATE_ALERT_MINIMUM_THRESHOLD)
                if total_count < min_threshold:
                    self.reset_trigger_counts()
                    metrics.incr('incidents.alert_rules.ignore_update_count_lower_than_min_threshold')
                    return None
        except KeyError:
            logger.exception('Received an update for a crash rate alert subscription, but no total sessions count was sent')
        aggregation_value_result: int = round((1 - aggregation_value) * 100, 3)
        return aggregation_value_result

    def get_crash_rate_alert_metrics_aggregation_value(self, subscription_update: SubscriptionUpdate) -> Optional[float]:
        if False:
            return 10
        'Handle both update formats. Once all subscriptions have been updated\n        to v2, we can remove v1 and replace this function with current v2.\n        '
        rows = subscription_update['values']['data']
        if BaseCrashRateMetricsEntitySubscription.is_crash_rate_format_v2(rows):
            version = 'v2'
            result = self._get_crash_rate_alert_metrics_aggregation_value_v2(subscription_update)
        else:
            version = 'v1'
            result = self._get_crash_rate_alert_metrics_aggregation_value_v1(subscription_update)
        metrics.incr('incidents.alert_rules.get_crash_rate_alert_metrics_aggregation_value', tags={'format': version}, sample_rate=1.0)
        return result

    def _get_crash_rate_alert_metrics_aggregation_value_v1(self, subscription_update: SubscriptionUpdate) -> Optional[float]:
        if False:
            print('Hello World!')
        '\n        Handles validation and extraction of Crash Rate Alerts subscription updates values over\n        metrics dataset.\n        The subscription update looks like\n        [\n            {\'project_id\': 8, \'tags[5]\': 6, \'value\': 2.0},\n            {\'project_id\': 8, \'tags[5]\': 13,\'value\': 1.0}\n        ]\n        where each entry represents a session status and the count of that specific session status.\n        As an example, `tags[5]` represents string `session.status`, while `tags[5]: 6` could\n        mean something like there are 2 sessions of status `crashed`. Likewise the other entry\n        represents the number of sessions started. In this method, we need to reverse match these\n        strings to end up with something that looks like\n        {"init": 2, "crashed": 4}\n        - `init` represents sessions or users sessions that were started, hence to get the crash\n        free percentage, we would need to divide number of crashed sessions by that number,\n        and subtract that value from 1. This is also used when CRASH_RATE_ALERT_MINIMUM_THRESHOLD is\n        set in the sense that if the minimum threshold is greater than the session count,\n        then the update is dropped. If the minimum threshold is not set then the total sessions\n        count is just ignored\n        - `crashed` represents the total sessions or user counts that crashed.\n        '
        (total_session_count, crash_count) = BaseCrashRateMetricsEntitySubscription.translate_sessions_tag_keys_and_values(data=subscription_update['values']['data'], org_id=self.subscription.project.organization.id)
        if total_session_count == 0:
            self.reset_trigger_counts()
            metrics.incr('incidents.alert_rules.ignore_update_no_session_data')
            return None
        if CRASH_RATE_ALERT_MINIMUM_THRESHOLD is not None:
            min_threshold = int(CRASH_RATE_ALERT_MINIMUM_THRESHOLD)
            if total_session_count < min_threshold:
                self.reset_trigger_counts()
                metrics.incr('incidents.alert_rules.ignore_update_count_lower_than_min_threshold')
                return None
        aggregation_value = round((1 - crash_count / total_session_count) * 100, 3)
        return aggregation_value

    def _get_crash_rate_alert_metrics_aggregation_value_v2(self, subscription_update: SubscriptionUpdate) -> Optional[float]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Handles validation and extraction of Crash Rate Alerts subscription updates values over\n        metrics dataset.\n        The subscription update looks like\n        [\n            {'project_id': 8, 'tags[5]': 6, 'count': 2.0, 'crashed': 1.0}\n        ]\n        - `count` represents sessions or users sessions that were started, hence to get the crash\n        free percentage, we would need to divide number of crashed sessions by that number,\n        and subtract that value from 1. This is also used when CRASH_RATE_ALERT_MINIMUM_THRESHOLD is\n        set in the sense that if the minimum threshold is greater than the session count,\n        then the update is dropped. If the minimum threshold is not set then the total sessions\n        count is just ignored\n        - `crashed` represents the total sessions or user counts that crashed.\n        "
        row = subscription_update['values']['data'][0]
        total_session_count = row['count']
        crash_count = row['crashed']
        if total_session_count == 0:
            self.reset_trigger_counts()
            metrics.incr('incidents.alert_rules.ignore_update_no_session_data')
            return None
        if CRASH_RATE_ALERT_MINIMUM_THRESHOLD is not None:
            min_threshold = int(CRASH_RATE_ALERT_MINIMUM_THRESHOLD)
            if total_session_count < min_threshold:
                self.reset_trigger_counts()
                metrics.incr('incidents.alert_rules.ignore_update_count_lower_than_min_threshold')
                return None
        aggregation_value: int = round((1 - crash_count / total_session_count) * 100, 3)
        return aggregation_value

    def get_aggregation_value(self, subscription_update: SubscriptionUpdate) -> Optional[float]:
        if False:
            for i in range(10):
                print('nop')
        if self.subscription.snuba_query.dataset == Dataset.Sessions.value:
            aggregation_value = self.get_crash_rate_alert_aggregation_value(subscription_update)
        elif self.subscription.snuba_query.dataset == Dataset.Metrics.value:
            aggregation_value = self.get_crash_rate_alert_metrics_aggregation_value(subscription_update)
        else:
            aggregation_value = list(subscription_update['values']['data'][0].values())[0]
            if aggregation_value is None:
                aggregation_value = 0
            if self.alert_rule.comparison_delta:
                aggregation_value = self.get_comparison_aggregation_value(subscription_update, aggregation_value)
        return aggregation_value

    def process_update(self, subscription_update: SubscriptionUpdate) -> None:
        if False:
            for i in range(10):
                print('nop')
        dataset = self.subscription.snuba_query.dataset
        try:
            self.subscription.project
        except Project.DoesNotExist:
            metrics.incr('incidents.alert_rules.ignore_deleted_project')
            return
        if dataset == 'events' and (not features.has('organizations:incidents', self.subscription.project.organization)):
            metrics.incr('incidents.alert_rules.ignore_update_missing_incidents')
            return
        elif dataset == 'transactions' and (not features.has('organizations:performance-view', self.subscription.project.organization)):
            metrics.incr('incidents.alert_rules.ignore_update_missing_incidents_performance')
            return
        if not hasattr(self, 'alert_rule'):
            metrics.incr('incidents.alert_rules.no_alert_rule_for_subscription')
            logger.error('Received an update for a subscription, but no associated alert rule exists')
            return
        if subscription_update['timestamp'] <= self.last_update:
            metrics.incr('incidents.alert_rules.skipping_already_processed_update')
            return
        self.last_update = subscription_update['timestamp']
        if len(subscription_update['values']['data']) > 1 and self.subscription.snuba_query.dataset != Dataset.Metrics.value:
            logger.warning('Subscription returned more than 1 row of data', extra={'subscription_id': self.subscription.id, 'dataset': self.subscription.snuba_query.dataset, 'snuba_subscription_id': self.subscription.subscription_id, 'result': subscription_update})
        aggregation_value = self.get_aggregation_value(subscription_update)
        if self.subscription.snuba_query.dataset == Dataset.Sessions.value:
            try:
                logger.info('subscription_processor.message', extra={'subscription_id': self.subscription.id, 'dataset': self.subscription.snuba_query.dataset, 'snuba_subscription_id': self.subscription.subscription_id, 'result': subscription_update, 'aggregation_value': aggregation_value})
            except Exception:
                logger.exception('Failed to log subscription results for session subscription')
        if aggregation_value is None:
            metrics.incr('incidents.alert_rules.skipping_update_invalid_aggregation_value')
            return
        (alert_operator, resolve_operator) = self.THRESHOLD_TYPE_OPERATORS[AlertRuleThresholdType(self.alert_rule.threshold_type)]
        fired_incident_triggers = []
        with transaction.atomic(router.db_for_write(AlertRule)):
            for trigger in self.triggers:
                if alert_operator(aggregation_value, trigger.alert_threshold) and (not self.check_trigger_status(trigger, TriggerStatus.ACTIVE)):
                    metrics.incr('incidents.alert_rules.threshold', tags={'type': 'alert'})
                    incident_trigger = self.trigger_alert_threshold(trigger, aggregation_value)
                    if incident_trigger is not None:
                        fired_incident_triggers.append(incident_trigger)
                else:
                    self.trigger_alert_counts[trigger.id] = 0
                if resolve_operator(aggregation_value, self.calculate_resolve_threshold(trigger)) and self.active_incident and self.check_trigger_status(trigger, TriggerStatus.ACTIVE):
                    metrics.incr('incidents.alert_rules.threshold', tags={'type': 'resolve'})
                    incident_trigger = self.trigger_resolve_threshold(trigger, aggregation_value)
                    if incident_trigger is not None:
                        fired_incident_triggers.append(incident_trigger)
                else:
                    self.trigger_resolve_counts[trigger.id] = 0
            if fired_incident_triggers:
                self.handle_trigger_actions(fired_incident_triggers, aggregation_value)
        self.update_alert_rule_stats()

    def calculate_event_date_from_update_date(self, update_date: datetime) -> datetime:
        if False:
            print('Hello World!')
        '\n        Calculates the date that an event actually happened based on the date that we\n        received the update. This takes into account time window and threshold period.\n        :return:\n        '
        update_date -= timedelta(seconds=self.alert_rule.snuba_query.time_window)
        return update_date - timedelta(seconds=self.alert_rule.snuba_query.resolution * (self.alert_rule.threshold_period - 1))

    def trigger_alert_threshold(self, trigger: AlertRuleTrigger, metric_value: float) -> IncidentTrigger | None:
        if False:
            return 10
        "\n        Called when a subscription update exceeds the value defined in the\n        `trigger.alert_threshold`, and the trigger hasn't already been activated.\n        Increments the count of how many times we've consecutively exceeded the threshold, and if\n        above the `threshold_period` defined in the alert rule then mark the trigger as\n        activated, and create an incident if there isn't already one.\n        :return:\n        "
        self.trigger_alert_counts[trigger.id] += 1
        if features.has('organizations:metric-alert-rate-limiting', self.subscription.project.organization):
            last_it = IncidentTrigger.objects.filter(alert_rule_trigger=trigger).order_by('-incident_id').select_related('incident').first()
            last_incident: Incident | None = last_it.incident if last_it else None
            last_incident_projects = [project.id for project in last_incident.projects.all()] if last_incident else []
            minutes_since_last_incident = (timezone.now() - last_incident.date_added).seconds / 60 if last_incident else None
            if last_incident and self.subscription.project.id in last_incident_projects and (minutes_since_last_incident <= 10):
                metrics.incr('incidents.alert_rules.hit_rate_limit', tags={'last_incident_id': last_incident.id, 'project_id': self.subscription.project.id, 'trigger_id': trigger.id})
                return None
        if self.trigger_alert_counts[trigger.id] >= self.alert_rule.threshold_period:
            metrics.incr('incidents.alert_rules.trigger', tags={'type': 'fire'})
            if not self.active_incident:
                detected_at = self.calculate_event_date_from_update_date(self.last_update)
                self.active_incident = create_incident(self.alert_rule.organization, IncidentType.ALERT_TRIGGERED, self.alert_rule.name, alert_rule=self.alert_rule, date_started=detected_at, date_detected=self.last_update, projects=[self.subscription.project])
            incident_trigger = self.incident_triggers.get(trigger.id)
            if incident_trigger:
                incident_trigger.status = TriggerStatus.ACTIVE.value
                incident_trigger.save()
            else:
                incident_trigger = IncidentTrigger.objects.create(incident=self.active_incident, alert_rule_trigger=trigger, status=TriggerStatus.ACTIVE.value)
            self.handle_incident_severity_update()
            self.incident_triggers[trigger.id] = incident_trigger
            self.trigger_alert_counts[trigger.id] = 0
            return incident_trigger
        else:
            return None

    def check_triggers_resolved(self) -> bool:
        if False:
            return 10
        '\n        Determines whether all triggers associated with the active incident are\n        resolved. A trigger is considered resolved if it is in the\n        `TriggerStatus.Resolved` state.\n        :return:\n        '
        for incident_trigger in self.incident_triggers.values():
            if incident_trigger.status != TriggerStatus.RESOLVED.value:
                return False
        return True

    def trigger_resolve_threshold(self, trigger: AlertRuleTrigger, metric_value: float) -> IncidentTrigger | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Called when a subscription update exceeds the trigger resolve threshold and the\n        trigger is currently ACTIVE.\n        :return:\n        '
        self.trigger_resolve_counts[trigger.id] += 1
        if self.trigger_resolve_counts[trigger.id] >= self.alert_rule.threshold_period:
            metrics.incr('incidents.alert_rules.trigger', tags={'type': 'resolve'})
            incident_trigger = self.incident_triggers[trigger.id]
            incident_trigger.status = TriggerStatus.RESOLVED.value
            incident_trigger.save()
            self.trigger_resolve_counts[trigger.id] = 0
            if self.check_triggers_resolved():
                update_incident_status(self.active_incident, IncidentStatus.CLOSED, status_method=IncidentStatusMethod.RULE_TRIGGERED, date_closed=self.calculate_event_date_from_update_date(self.last_update))
                self.active_incident = None
                self.incident_triggers.clear()
            else:
                self.handle_incident_severity_update()
            return incident_trigger
        else:
            return None

    def handle_trigger_actions(self, incident_triggers: List[IncidentTrigger], metric_value: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        actions = deduplicate_trigger_actions(triggers=deepcopy(self.triggers))
        incident_trigger = incident_triggers[0]
        method = 'fire' if incident_trigger.status == TriggerStatus.ACTIVE.value else 'resolve'
        try:
            incident = Incident.objects.get(id=incident_trigger.incident_id)
        except Incident.DoesNotExist:
            metrics.incr('incidents.alert_rules.action.skipping_missing_incident')
            return
        incident_activities = IncidentActivity.objects.filter(incident=incident).values_list('value', flat=True)
        past_statuses = {int(value) for value in incident_activities.distinct() if value is not None}
        critical_actions = []
        warning_actions = []
        for action in actions:
            if action.alert_rule_trigger.label == CRITICAL_TRIGGER_LABEL:
                critical_actions.append(action)
            else:
                warning_actions.append(action)
        actions_to_fire = []
        new_status = IncidentStatus.CLOSED.value
        if method == 'resolve':
            if incident.status != IncidentStatus.CLOSED.value:
                actions_to_fire = actions
                new_status = IncidentStatus.WARNING.value
            elif IncidentStatus.CRITICAL.value in past_statuses:
                actions_to_fire = actions
                new_status = IncidentStatus.CLOSED.value
            else:
                actions_to_fire = warning_actions
                new_status = IncidentStatus.CLOSED.value
        elif incident.status == IncidentStatus.CRITICAL.value:
            actions_to_fire = actions
            new_status = IncidentStatus.CRITICAL.value
        else:
            actions_to_fire = warning_actions
            new_status = IncidentStatus.WARNING.value
        for action in actions_to_fire:
            transaction.on_commit(handle_trigger_action.s(action_id=action.id, incident_id=incident.id, project_id=self.subscription.project_id, method=method, new_status=new_status, metric_value=metric_value).delay, router.db_for_write(AlertRule))

    def handle_incident_severity_update(self) -> None:
        if False:
            print('Hello World!')
        if self.active_incident:
            active_incident_triggers = IncidentTrigger.objects.filter(incident=self.active_incident, status=TriggerStatus.ACTIVE.value)
            severity = None
            for active_incident_trigger in active_incident_triggers:
                trigger = active_incident_trigger.alert_rule_trigger
                if trigger.label == CRITICAL_TRIGGER_LABEL:
                    severity = IncidentStatus.CRITICAL
                    break
                elif trigger.label == WARNING_TRIGGER_LABEL:
                    severity = IncidentStatus.WARNING
            if severity:
                update_incident_status(self.active_incident, severity, status_method=IncidentStatusMethod.RULE_TRIGGERED)

    def update_alert_rule_stats(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Updates stats about the alert rule, if they're changed.\n        :return:\n        "
        updated_trigger_alert_counts = {trigger_id: alert_count for (trigger_id, alert_count) in self.trigger_alert_counts.items() if alert_count != self.orig_trigger_alert_counts[trigger_id]}
        updated_trigger_resolve_counts = {trigger_id: alert_count for (trigger_id, alert_count) in self.trigger_resolve_counts.items() if alert_count != self.orig_trigger_resolve_counts[trigger_id]}
        update_alert_rule_stats(self.alert_rule, self.subscription, self.last_update, updated_trigger_alert_counts, updated_trigger_resolve_counts)

def build_alert_rule_stat_keys(alert_rule: AlertRule, subscription: QuerySubscription) -> List[str]:
    if False:
        i = 10
        return i + 15
    '\n    Builds keys for fetching stats about alert rules\n    :return: A list containing the alert rule stat keys\n    '
    key_base = ALERT_RULE_BASE_KEY % (alert_rule.id, subscription.project_id)
    return [ALERT_RULE_BASE_STAT_KEY % (key_base, stat_key) for stat_key in ALERT_RULE_STAT_KEYS]

def build_trigger_stat_keys(alert_rule: AlertRule, subscription: QuerySubscription, triggers: List[AlertRuleTrigger]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Builds keys for fetching stats about triggers\n    :return: A list containing the alert rule trigger stat keys\n    '
    return [build_alert_rule_trigger_stat_key(alert_rule.id, subscription.project_id, trigger.id, stat_key) for trigger in triggers for stat_key in ALERT_RULE_TRIGGER_STAT_KEYS]

def build_alert_rule_trigger_stat_key(alert_rule_id: int, project_id: int, trigger_id: int, stat_key: str) -> str:
    if False:
        i = 10
        return i + 15
    key_base = ALERT_RULE_BASE_KEY % (alert_rule_id, project_id)
    return ALERT_RULE_BASE_TRIGGER_STAT_KEY % (key_base, trigger_id, stat_key)

def partition(iterable: Sequence[T], n: int) -> Sequence[Sequence[T]]:
    if False:
        return 10
    "\n    Partitions an iterable into tuples of size n. Expects the iterable length to be a\n    multiple of n.\n    partition('ABCDEF', 3) --> [('A', 'B', 'C'), ('D', 'E', 'F')]\n    "
    assert len(iterable) % n == 0
    args = [iter(iterable)] * n
    return cast(Sequence[Sequence[T]], zip(*args))

def get_alert_rule_stats(alert_rule: AlertRule, subscription: QuerySubscription, triggers: List[AlertRuleTrigger]) -> Tuple[datetime, Dict[str, int], Dict[str, int]]:
    if False:
        print('Hello World!')
    '\n    Fetches stats about the alert rule, specific to the current subscription\n    :return: A tuple containing the stats about the alert rule and subscription.\n     - last_update: Int representing the timestamp it was last updated\n     - trigger_alert_counts: A dict of trigger alert counts, where the key is the\n       trigger id, and the value is an int representing how many consecutive times we\n       have triggered the alert threshold\n     - trigger_resolve_counts: A dict of trigger resolve counts, where the key is the\n       trigger id, and the value is an int representing how many consecutive times we\n       have triggered the resolve threshold\n    '
    alert_rule_keys = build_alert_rule_stat_keys(alert_rule, subscription)
    trigger_keys = build_trigger_stat_keys(alert_rule, subscription, triggers)
    results = get_redis_client().mget(alert_rule_keys + trigger_keys)
    results = tuple((0 if result is None else int(result) for result in results))
    last_update = to_datetime(results[0])
    trigger_results = results[1:]
    trigger_alert_counts = {}
    trigger_resolve_counts = {}
    for (trigger, trigger_result) in zip(triggers, partition(trigger_results, len(ALERT_RULE_TRIGGER_STAT_KEYS))):
        trigger_alert_counts[trigger.id] = trigger_result[0]
        trigger_resolve_counts[trigger.id] = trigger_result[1]
    return (last_update, trigger_alert_counts, trigger_resolve_counts)

def update_alert_rule_stats(alert_rule: AlertRule, subscription: QuerySubscription, last_update: datetime, alert_counts: Dict[int, int], resolve_counts: Dict[int, int]) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    Updates stats about the alert rule, subscription and triggers if they've changed.\n    "
    pipeline = get_redis_client().pipeline()
    counts_with_stat_keys = zip(ALERT_RULE_TRIGGER_STAT_KEYS, (alert_counts, resolve_counts))
    for (stat_key, trigger_counts) in counts_with_stat_keys:
        for (trigger_id, alert_count) in trigger_counts.items():
            pipeline.set(build_alert_rule_trigger_stat_key(alert_rule.id, subscription.project_id, trigger_id, stat_key), alert_count, ex=REDIS_TTL)
    last_update_key = build_alert_rule_stat_keys(alert_rule, subscription)[0]
    pipeline.set(last_update_key, int(to_timestamp(last_update)), ex=REDIS_TTL)
    pipeline.execute()

def get_redis_client() -> RetryingRedisCluster:
    if False:
        i = 10
        return i + 15
    cluster_key = settings.SENTRY_INCIDENT_RULES_REDIS_CLUSTER
    return redis.redis_clusters.get(cluster_key)
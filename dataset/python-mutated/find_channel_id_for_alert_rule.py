from __future__ import annotations
import logging
from typing import Any
from rest_framework import serializers
from sentry.auth.access import SystemAccess
from sentry.incidents.logic import ChannelLookupTimeoutError, InvalidTriggerActionError, get_slack_channel_ids
from sentry.incidents.models import AlertRule
from sentry.incidents.serializers import AlertRuleSerializer
from sentry.integrations.slack.utils import SLACK_RATE_LIMITED_MESSAGE, RedisRuleStatus
from sentry.models.organization import Organization
from sentry.services.hybrid_cloud.user import RpcUser
from sentry.services.hybrid_cloud.user.service import user_service
from sentry.shared_integrations.exceptions import ApiRateLimitedError
from sentry.silo import SiloMode
from sentry.tasks.base import instrumented_task
logger = logging.getLogger('sentry.integrations.slack.tasks')

@instrumented_task(name='sentry.integrations.slack.search_channel_id_metric_alerts', queue='integrations', silo_mode=SiloMode.REGION)
def find_channel_id_for_alert_rule(organization_id: int, uuid: str, data: Any, alert_rule_id: int | None=None, user_id: int | None=None) -> None:
    if False:
        while True:
            i = 10
    redis_rule_status = RedisRuleStatus(uuid)
    try:
        organization = Organization.objects.get(id=organization_id)
    except Organization.DoesNotExist:
        redis_rule_status.set_value('failed')
        return
    user: RpcUser | None = None
    if user_id:
        user = user_service.get_user(user_id=user_id)
    alert_rule = None
    if alert_rule_id:
        try:
            alert_rule = AlertRule.objects.get(organization_id=organization_id, id=alert_rule_id)
        except AlertRule.DoesNotExist:
            redis_rule_status.set_value('failed')
            return
    try:
        mapped_ids = get_slack_channel_ids(organization, user, data)
    except (serializers.ValidationError, ChannelLookupTimeoutError, InvalidTriggerActionError) as e:
        logger.info('get_slack_channel_ids.failed', extra={'exception': e})
        redis_rule_status.set_value('failed')
        return
    except ApiRateLimitedError as e:
        logger.info('get_slack_channel_ids.rate_limited', extra={'exception': e})
        redis_rule_status.set_value('failed', None, SLACK_RATE_LIMITED_MESSAGE)
        return
    for trigger in data['triggers']:
        for action in trigger['actions']:
            if action['type'] == 'slack':
                if action['targetIdentifier'] in mapped_ids:
                    action['input_channel_id'] = mapped_ids[action['targetIdentifier']]
                else:
                    redis_rule_status.set_value('failed')
                    return
    serializer = AlertRuleSerializer(context={'organization': organization, 'access': SystemAccess(), 'user': user, 'use_async_lookup': True, 'validate_channel_id': False}, data=data, instance=alert_rule)
    if serializer.is_valid():
        try:
            alert_rule = serializer.save()
            redis_rule_status.set_value('success', alert_rule.id)
            return
        except (serializers.ValidationError, ChannelLookupTimeoutError):
            redis_rule_status.set_value('failed')
            return
    redis_rule_status.set_value('failed')
    return
import logging
from typing import Any, Optional, Sequence
from sentry.incidents.models import AlertRuleTriggerAction
from sentry.integrations.slack.utils import SLACK_RATE_LIMITED_MESSAGE, RedisRuleStatus, get_channel_id_with_timeout, strip_channel_name
from sentry.mediators.project_rules.creator import Creator
from sentry.mediators.project_rules.updater import Updater
from sentry.models.project import Project
from sentry.models.rule import Rule, RuleActivity, RuleActivityType
from sentry.services.hybrid_cloud.integration import integration_service
from sentry.shared_integrations.exceptions import ApiRateLimitedError, DuplicateDisplayNameError
from sentry.silo import SiloMode
from sentry.tasks.base import instrumented_task
logger = logging.getLogger('sentry.integrations.slack.tasks')

@instrumented_task(name='sentry.integrations.slack.search_channel_id', queue='integrations', silo_mode=SiloMode.REGION)
def find_channel_id_for_rule(project: Project, actions: Sequence[AlertRuleTriggerAction], uuid: str, rule_id: Optional[int]=None, user_id: Optional[int]=None, **kwargs: Any) -> None:
    if False:
        while True:
            i = 10
    redis_rule_status = RedisRuleStatus(uuid)
    try:
        project = Project.objects.get(id=project.id)
    except Project.DoesNotExist:
        redis_rule_status.set_value('failed')
        return
    organization = project.organization
    integration_id: Optional[int] = None
    channel_name: Optional[str] = None
    for action in actions:
        if action.get('workspace') and action.get('channel'):
            integration_id = action['workspace']
            channel_name = strip_channel_name(action['channel'])
            break
    integrations = integration_service.get_integrations(organization_id=organization.id, providers=['slack'], integration_ids=[integration_id])
    if not integrations:
        redis_rule_status.set_value('failed')
        return
    integration = integrations[0]
    logger.info('rule.slack.search_channel_id', extra={'integration_id': integration.id, 'organization_id': organization.id, 'rule_id': rule_id})
    try:
        (prefix, item_id, _timed_out) = get_channel_id_with_timeout(integration, channel_name, 3 * 60)
    except DuplicateDisplayNameError:
        item_id = None
        prefix = ''
    except ApiRateLimitedError:
        redis_rule_status.set_value('failed', None, SLACK_RATE_LIMITED_MESSAGE)
        return
    if item_id:
        for action in actions:
            if action.get('channel') and strip_channel_name(action.get('channel')) == channel_name:
                action['channel'] = prefix + channel_name
                action['channel_id'] = item_id
                break
        kwargs['actions'] = actions
        kwargs['project'] = project
        if rule_id:
            rule = Rule.objects.get(id=rule_id)
            rule = Updater.run(rule=rule, pending_save=False, **kwargs)
        else:
            rule = Creator.run(pending_save=False, **kwargs)
            if user_id:
                RuleActivity.objects.create(rule=rule, user_id=user_id, type=RuleActivityType.CREATED.value)
        redis_rule_status.set_value('success', rule.id)
        return
    redis_rule_status.set_value('failed')
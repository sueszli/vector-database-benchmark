from __future__ import annotations
import logging
from datetime import datetime, timezone
from django.db.models import F
from django.utils import timezone as django_timezone
from sentry import analytics
from sentry.models.organization import Organization
from sentry.models.organizationonboardingtask import OnboardingTask, OnboardingTaskStatus, OrganizationOnboardingTask
from sentry.models.project import Project
from sentry.onboarding_tasks import try_mark_onboarding_complete
from sentry.plugins.bases.issue import IssueTrackingPlugin
from sentry.plugins.bases.issue2 import IssueTrackingPlugin2
from sentry.services.hybrid_cloud.integration import RpcIntegration, integration_service
from sentry.services.hybrid_cloud.user import RpcUser
from sentry.signals import alert_rule_created, cron_monitor_created, event_processed, first_cron_checkin_received, first_cron_monitor_created, first_event_pending, first_event_received, first_event_with_minified_stack_trace_received, first_feedback_received, first_profile_received, first_replay_received, first_transaction_received, integration_added, issue_tracker_used, member_invited, member_joined, plugin_enabled, project_created, transaction_processed
from sentry.utils.event import has_event_minified_stack_trace
from sentry.utils.javascript import has_sourcemap
from sentry.utils.safe import get_path
logger = logging.getLogger('sentry')
START_DATE_TRACKING_FIRST_EVENT_WITH_MINIFIED_STACK_TRACE_PER_PROJ = datetime(2022, 12, 14, tzinfo=timezone.utc)

@project_created.connect(weak=False)
def record_new_project(project, user=None, user_id=None, **kwargs):
    if False:
        while True:
            i = 10
    if user_id is not None:
        default_user_id = user_id
    elif user.is_authenticated:
        user_id = default_user_id = user.id
    else:
        user_id = None
        try:
            default_user_id = Organization.objects.get(id=project.organization_id).get_default_owner().id
        except IndexError:
            logger.warning('Cannot initiate onboarding for organization (%s) due to missing owners', project.organization_id)
            return
    analytics.record('project.created', user_id=user_id, default_user_id=default_user_id, organization_id=project.organization_id, project_id=project.id, platform=project.platform)
    success = OrganizationOnboardingTask.objects.record(organization_id=project.organization_id, task=OnboardingTask.FIRST_PROJECT, user_id=user_id, status=OnboardingTaskStatus.COMPLETE, project_id=project.id)
    if not success:
        OrganizationOnboardingTask.objects.record(organization_id=project.organization_id, task=OnboardingTask.SECOND_PLATFORM, user_id=user_id, status=OnboardingTaskStatus.PENDING, project_id=project.id)

@first_event_pending.connect(weak=False)
def record_raven_installed(project, user, **kwargs):
    if False:
        i = 10
        return i + 15
    OrganizationOnboardingTask.objects.record(organization_id=project.organization_id, task=OnboardingTask.FIRST_EVENT, status=OnboardingTaskStatus.PENDING, user_id=user.id if user else None, project_id=project.id)

@first_event_received.connect(weak=False)
def record_first_event(project, event, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Requires up to 2 database calls, but should only run with the first event in\n    any project, so performance should not be a huge bottleneck.\n    '
    (rows_affected, created) = OrganizationOnboardingTask.objects.create_or_update(organization_id=project.organization_id, task=OnboardingTask.FIRST_EVENT, status=OnboardingTaskStatus.PENDING, values={'status': OnboardingTaskStatus.COMPLETE, 'project_id': project.id, 'date_completed': project.first_event, 'data': {'platform': event.platform}})
    try:
        user: RpcUser = Organization.objects.get(id=project.organization_id).get_default_owner()
    except IndexError:
        logger.warning('Cannot record first event for organization (%s) due to missing owners', project.organization_id)
        return
    analytics.record('first_event_for_project.sent', user_id=user.id if user else None, organization_id=project.organization_id, project_id=project.id, platform=event.platform, project_platform=project.platform, url=dict(event.tags).get('url', None), has_minified_stack_trace=has_event_minified_stack_trace(event), sdk_name=get_path(event, 'sdk', 'name'))
    if rows_affected or created:
        analytics.record('first_event.sent', user_id=user.id if user else None, organization_id=project.organization_id, project_id=project.id, platform=event.platform, project_platform=project.platform)
        return
    try:
        oot = OrganizationOnboardingTask.objects.filter(organization_id=project.organization_id, task=OnboardingTask.FIRST_EVENT)[0]
    except IndexError:
        return
    if oot.project_id != project.id:
        (rows_affected, created) = OrganizationOnboardingTask.objects.create_or_update(organization_id=project.organization_id, task=OnboardingTask.SECOND_PLATFORM, status=OnboardingTaskStatus.PENDING, values={'status': OnboardingTaskStatus.COMPLETE, 'project_id': project.id, 'date_completed': project.first_event, 'data': {'platform': event.platform}})
        if rows_affected or created:
            analytics.record('second_platform.added', user_id=user.id if user else None, organization_id=project.organization_id, project_id=project.id, platform=event.platform)

@first_transaction_received.connect(weak=False)
def record_first_transaction(project, event, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    project.update(flags=F('flags').bitor(Project.flags.has_transactions))
    OrganizationOnboardingTask.objects.record(organization_id=project.organization_id, task=OnboardingTask.FIRST_TRANSACTION, status=OnboardingTaskStatus.COMPLETE, date_completed=event.datetime)
    try:
        default_user_id = project.organization.get_default_owner().id
    except IndexError:
        default_user_id = None
    analytics.record('first_transaction.sent', default_user_id=default_user_id, organization_id=project.organization_id, project_id=project.id, platform=project.platform)

@first_profile_received.connect(weak=False)
def record_first_profile(project, **kwargs):
    if False:
        i = 10
        return i + 15
    project.update(flags=F('flags').bitor(Project.flags.has_profiles))
    analytics.record('first_profile.sent', user_id=project.organization.default_owner_id, organization_id=project.organization_id, project_id=project.id, platform=project.platform)

@first_replay_received.connect(weak=False)
def record_first_replay(project, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    project.update(flags=F('flags').bitor(Project.flags.has_replays))
    success = OrganizationOnboardingTask.objects.record(organization_id=project.organization_id, task=OnboardingTask.SESSION_REPLAY, status=OnboardingTaskStatus.COMPLETE, date_completed=django_timezone.now())
    if success:
        analytics.record('first_replay.sent', user_id=project.organization.default_owner_id, organization_id=project.organization_id, project_id=project.id, platform=project.platform)
        try_mark_onboarding_complete(project.organization_id)

@first_feedback_received.connect(weak=False)
def record_first_feedback(project, **kwargs):
    if False:
        print('Hello World!')
    project.update(flags=F('flags').bitor(Project.flags.has_feedbacks))
    analytics.record('first_feedback.sent', user_id=project.organization.default_owner_id, organization_id=project.organization_id, project_id=project.id, platform=project.platform)

@first_cron_monitor_created.connect(weak=False)
def record_first_cron_monitor(project, user, from_upsert, **kwargs):
    if False:
        while True:
            i = 10
    updated = project.update(flags=F('flags').bitor(Project.flags.has_cron_monitors))
    if updated:
        analytics.record('first_cron_monitor.created', user_id=user.id if user else project.organization.default_owner_id, organization_id=project.organization_id, project_id=project.id, from_upsert=from_upsert)

@cron_monitor_created.connect(weak=False)
def record_cron_monitor_created(project, user, from_upsert, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    analytics.record('cron_monitor.created', user_id=user.id if user else project.organization.default_owner_id, organization_id=project.organization_id, project_id=project.id, from_upsert=from_upsert)

@first_cron_checkin_received.connect(weak=False)
def record_first_cron_checkin(project, monitor_id, **kwargs):
    if False:
        i = 10
        return i + 15
    project.update(flags=F('flags').bitor(Project.flags.has_cron_checkins))
    analytics.record('first_cron_checkin.sent', user_id=project.organization.default_owner_id, organization_id=project.organization_id, project_id=project.id, monitor_id=monitor_id)

@member_invited.connect(weak=False)
def record_member_invited(member, user, **kwargs):
    if False:
        i = 10
        return i + 15
    OrganizationOnboardingTask.objects.record(organization_id=member.organization_id, task=OnboardingTask.INVITE_MEMBER, user_id=user.id if user else None, status=OnboardingTaskStatus.PENDING, data={'invited_member_id': member.id})
    analytics.record('member.invited', invited_member_id=member.id, inviter_user_id=user.id if user else None, organization_id=member.organization_id, referrer=kwargs.get('referrer'))

@member_joined.connect(weak=False)
def record_member_joined(organization_id: int, organization_member_id: int, **kwargs):
    if False:
        i = 10
        return i + 15
    (rows_affected, created) = OrganizationOnboardingTask.objects.create_or_update(organization_id=organization_id, task=OnboardingTask.INVITE_MEMBER, status=OnboardingTaskStatus.PENDING, values={'status': OnboardingTaskStatus.COMPLETE, 'date_completed': django_timezone.now(), 'data': {'invited_member_id': organization_member_id}})
    if created or rows_affected:
        try_mark_onboarding_complete(organization_id)

def record_release_received(project, event, **kwargs):
    if False:
        i = 10
        return i + 15
    if not event.get_tag('sentry:release'):
        return
    success = OrganizationOnboardingTask.objects.record(organization_id=project.organization_id, task=OnboardingTask.RELEASE_TRACKING, status=OnboardingTaskStatus.COMPLETE, project_id=project.id)
    if success:
        try:
            user: RpcUser = Organization.objects.get(id=project.organization_id).get_default_owner()
        except IndexError:
            logger.warning('Cannot record release received for organization (%s) due to missing owners', project.organization_id)
            return
        analytics.record('first_release_tag.sent', user_id=user.id if user else None, project_id=project.id, organization_id=project.organization_id)
        try_mark_onboarding_complete(project.organization_id)
event_processed.connect(record_release_received, weak=False)
transaction_processed.connect(record_release_received, weak=False)

def record_user_context_received(project, event, **kwargs):
    if False:
        i = 10
        return i + 15
    user_context = event.data.get('user')
    if not user_context:
        return
    elif list(user_context.keys()) != ['ip_address']:
        success = OrganizationOnboardingTask.objects.record(organization_id=project.organization_id, task=OnboardingTask.USER_CONTEXT, status=OnboardingTaskStatus.COMPLETE, project_id=project.id)
        if success:
            try:
                user: RpcUser = Organization.objects.get(id=project.organization_id).get_default_owner()
            except IndexError:
                logger.warning('Cannot record user context received for organization (%s) due to missing owners', project.organization_id)
                return
            analytics.record('first_user_context.sent', user_id=user.id if user else None, organization_id=project.organization_id, project_id=project.id)
            try_mark_onboarding_complete(project.organization_id)
event_processed.connect(record_user_context_received, weak=False)

@first_event_with_minified_stack_trace_received.connect(weak=False)
def record_event_with_first_minified_stack_trace_for_project(project, event, **kwargs):
    if False:
        return 10
    try:
        user: RpcUser = Organization.objects.get(id=project.organization_id).get_default_owner()
    except IndexError:
        logger.warning('Cannot record first event for organization (%s) due to missing owners', project.organization_id)
        return
    if not project.flags.has_minified_stack_trace:
        affected = Project.objects.filter(id=project.id, flags=F('flags').bitand(~Project.flags.has_minified_stack_trace)).update(flags=F('flags').bitor(Project.flags.has_minified_stack_trace))
        if project.date_added > START_DATE_TRACKING_FIRST_EVENT_WITH_MINIFIED_STACK_TRACE_PER_PROJ and affected > 0:
            analytics.record('first_event_with_minified_stack_trace_for_project.sent', user_id=user.id if user else None, organization_id=project.organization_id, project_id=project.id, platform=event.platform, project_platform=project.platform, url=dict(event.tags).get('url', None))
transaction_processed.connect(record_user_context_received, weak=False)

@event_processed.connect(weak=False)
def record_sourcemaps_received(project, event, **kwargs):
    if False:
        while True:
            i = 10
    if not has_sourcemap(event):
        return
    success = OrganizationOnboardingTask.objects.record(organization_id=project.organization_id, task=OnboardingTask.SOURCEMAPS, status=OnboardingTaskStatus.COMPLETE, project_id=project.id)
    if success:
        try:
            user: RpcUser = Organization.objects.get(id=project.organization_id).get_default_owner()
        except IndexError:
            logger.warning('Cannot record sourcemaps received for organization (%s) due to missing owners', project.organization_id)
            return
        analytics.record('first_sourcemaps.sent', user_id=user.id if user else None, organization_id=project.organization_id, project_id=project.id)
        try_mark_onboarding_complete(project.organization_id)

@plugin_enabled.connect(weak=False)
def record_plugin_enabled(plugin, project, user, **kwargs):
    if False:
        print('Hello World!')
    if isinstance(plugin, IssueTrackingPlugin) or isinstance(plugin, IssueTrackingPlugin2):
        task = OnboardingTask.ISSUE_TRACKER
        status = OnboardingTaskStatus.PENDING
    else:
        return
    success = OrganizationOnboardingTask.objects.record(organization_id=project.organization_id, task=task, status=status, user_id=user.id if user else None, project_id=project.id, data={'plugin': plugin.slug})
    if success:
        try_mark_onboarding_complete(project.organization_id)
    analytics.record('plugin.enabled', user_id=user.id if user else None, organization_id=project.organization_id, project_id=project.id, plugin=plugin.slug)

@alert_rule_created.connect(weak=False)
def record_alert_rule_created(user, project, rule, rule_type, **kwargs):
    if False:
        print('Hello World!')
    task = OnboardingTask.METRIC_ALERT if rule_type == 'metric' else OnboardingTask.ALERT_RULE
    (rows_affected, created) = OrganizationOnboardingTask.objects.create_or_update(organization_id=project.organization_id, task=task, values={'status': OnboardingTaskStatus.COMPLETE, 'user_id': user.id if user else None, 'project_id': project.id, 'date_completed': django_timezone.now()})
    if rows_affected or created:
        try_mark_onboarding_complete(project.organization_id)

@issue_tracker_used.connect(weak=False)
def record_issue_tracker_used(plugin, project, user, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    (rows_affected, created) = OrganizationOnboardingTask.objects.create_or_update(organization_id=project.organization_id, task=OnboardingTask.ISSUE_TRACKER, status=OnboardingTaskStatus.PENDING, values={'status': OnboardingTaskStatus.COMPLETE, 'user_id': user.id, 'project_id': project.id, 'date_completed': django_timezone.now(), 'data': {'plugin': plugin.slug}})
    if rows_affected or created:
        try_mark_onboarding_complete(project.organization_id)
    if user and user.is_authenticated:
        user_id = default_user_id = user.id
    else:
        user_id = None
        try:
            default_user_id = project.organization.get_default_owner().id
        except IndexError:
            logger.warning('Cannot record issue tracker used for organization (%s) due to missing owners', project.organization_id)
            return
    analytics.record('issue_tracker.used', user_id=user_id, default_user_id=default_user_id, organization_id=project.organization_id, project_id=project.id, issue_tracker=plugin.slug)

@integration_added.connect(weak=False)
def record_integration_added(integration_id: int, organization_id: int, user_id: int | None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    integration: RpcIntegration | None = integration_service.get_integration(integration_id=integration_id)
    if integration is None:
        return
    task = OrganizationOnboardingTask.objects.filter(organization_id=organization_id, task=OnboardingTask.INTEGRATIONS).first()
    if task:
        providers = task.data.get('providers', [])
        if integration.provider not in providers:
            providers.append(integration.provider)
        task.data['providers'] = providers
        if task.status != OnboardingTaskStatus.COMPLETE:
            task.status = OnboardingTaskStatus.COMPLETE
            task.user_id = user_id
            task.date_completed = django_timezone.now()
        task.save()
    else:
        task = OrganizationOnboardingTask.objects.create(organization_id=organization_id, task=OnboardingTask.INTEGRATIONS, status=OnboardingTaskStatus.COMPLETE, data={'providers': [integration.provider]})
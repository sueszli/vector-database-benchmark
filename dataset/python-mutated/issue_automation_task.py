import json
from datetime import timedelta
from django.utils import timezone
from django.db.models import Q
from django.conf import settings
from celery import shared_task
from sentry_sdk import capture_exception
from plane.db.models import Issue, Project, State
from plane.bgtasks.issue_activites_task import issue_activity

@shared_task
def archive_and_close_old_issues():
    if False:
        return 10
    archive_old_issues()
    close_old_issues()

def archive_old_issues():
    if False:
        return 10
    try:
        projects = Project.objects.filter(archive_in__gt=0)
        for project in projects:
            project_id = project.id
            archive_in = project.archive_in
            issues = Issue.issue_objects.filter(Q(project=project_id, archived_at__isnull=True, updated_at__lte=timezone.now() - timedelta(days=archive_in * 30), state__group__in=['completed', 'cancelled']), Q(issue_cycle__isnull=True) | Q(issue_cycle__cycle__end_date__lt=timezone.now().date()) & Q(issue_cycle__isnull=False), Q(issue_module__isnull=True) | Q(issue_module__module__target_date__lt=timezone.now().date()) & Q(issue_module__isnull=False)).filter(Q(issue_inbox__status=1) | Q(issue_inbox__status=-1) | Q(issue_inbox__status=2) | Q(issue_inbox__isnull=True))
            if issues:
                archive_at = timezone.now().date()
                issues_to_update = []
                for issue in issues:
                    issue.archived_at = archive_at
                    issues_to_update.append(issue)
                if issues_to_update:
                    Issue.objects.bulk_update(issues_to_update, ['archived_at'], batch_size=100)
                    _ = [issue_activity.delay(type='issue.activity.updated', requested_data=json.dumps({'archived_at': str(archive_at)}), actor_id=str(project.created_by_id), issue_id=issue.id, project_id=project_id, current_instance=json.dumps({'archived_at': None}), subscriber=False, epoch=int(timezone.now().timestamp())) for issue in issues_to_update]
        return
    except Exception as e:
        if settings.DEBUG:
            print(e)
        capture_exception(e)
        return

def close_old_issues():
    if False:
        for i in range(10):
            print('nop')
    try:
        projects = Project.objects.filter(close_in__gt=0).select_related('default_state')
        for project in projects:
            project_id = project.id
            close_in = project.close_in
            issues = Issue.issue_objects.filter(Q(project=project_id, archived_at__isnull=True, updated_at__lte=timezone.now() - timedelta(days=close_in * 30), state__group__in=['backlog', 'unstarted', 'started']), Q(issue_cycle__isnull=True) | Q(issue_cycle__cycle__end_date__lt=timezone.now().date()) & Q(issue_cycle__isnull=False), Q(issue_module__isnull=True) | Q(issue_module__module__target_date__lt=timezone.now().date()) & Q(issue_module__isnull=False)).filter(Q(issue_inbox__status=1) | Q(issue_inbox__status=-1) | Q(issue_inbox__status=2) | Q(issue_inbox__isnull=True))
            if issues:
                if project.default_state is None:
                    close_state = State.objects.filter(group='cancelled').first()
                else:
                    close_state = project.default_state
                issues_to_update = []
                for issue in issues:
                    issue.state = close_state
                    issues_to_update.append(issue)
                if issues_to_update:
                    Issue.objects.bulk_update(issues_to_update, ['state'], batch_size=100)
                    [issue_activity.delay(type='issue.activity.updated', requested_data=json.dumps({'closed_to': str(issue.state_id)}), actor_id=str(project.created_by_id), issue_id=issue.id, project_id=project_id, current_instance=None, subscriber=False, epoch=int(timezone.now().timestamp())) for issue in issues_to_update]
        return
    except Exception as e:
        if settings.DEBUG:
            print(e)
        capture_exception(e)
        return
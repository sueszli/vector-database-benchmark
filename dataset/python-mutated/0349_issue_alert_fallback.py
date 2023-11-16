import logging
from enum import Enum
from django.db import migrations, transaction
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapper, RangeQuerySetWrapperWithProgressBar

class RuleStatus(Enum):
    ACTIVE = 0

class ProjectStatus(Enum):
    ACTIVE = 0
    DISABLED = 1

class OrganizationStatus(Enum):
    ACTIVE = 0

def set_issue_alert_fallback(rule, fallthrough_choice):
    if False:
        return 10
    actions = rule.data.get('actions', [])
    rule_changed = False
    for action in actions:
        id = action.get('id')
        target_type = action.get('targetType')
        if id == 'sentry.mail.actions.NotifyEmailAction' and target_type == 'IssueOwners':
            if 'fallthroughType' not in action:
                action.update({'fallthroughType': fallthrough_choice})
            rule_changed = True
    if rule_changed:
        rule.data['actions'] = actions
        rule.save()

def migrate_project_ownership_to_issue_alert_fallback(project, ProjectOwnership, Rule):
    if False:
        while True:
            i = 10
    with transaction.atomic('default'):
        fallthrough_choice = None
        try:
            ownership = ProjectOwnership.objects.get(project=project)
            fallthrough_choice = 'AllMembers' if ownership and ownership.fallthrough else 'NoOne'
        except ProjectOwnership.DoesNotExist:
            fallthrough_choice = 'ActiveMembers'
        for rule in Rule.objects.filter(project=project, status=RuleStatus.ACTIVE.value):
            set_issue_alert_fallback(rule, fallthrough_choice)

def migrate_to_issue_alert_fallback(apps, schema_editor):
    if False:
        while True:
            i = 10
    Project = apps.get_model('sentry', 'Project')
    ProjectOwnership = apps.get_model('sentry', 'ProjectOwnership')
    Organization = apps.get_model('sentry', 'Organization')
    Rule = apps.get_model('sentry', 'Rule')
    for org in RangeQuerySetWrapperWithProgressBar(Organization.objects.filter(status=OrganizationStatus.ACTIVE.value)):
        for project in RangeQuerySetWrapper(Project.objects.filter(organization=org, status__in=[ProjectStatus.ACTIVE.value, ProjectStatus.DISABLED.value])):
            try:
                migrate_project_ownership_to_issue_alert_fallback(project, ProjectOwnership, Rule)
            except Exception:
                logging.exception(f'Error migrating project {project.id}')

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0348_add_outbox_and_tombstone_tables')]
    operations = [migrations.RunPython(migrate_to_issue_alert_fallback, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_rule']})]
from django.db import migrations
from sentry.monitors.models import MonitorStatus
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar
TERMINAL_STATES = [MonitorStatus.PENDING_DELETION, MonitorStatus.DELETION_IN_PROGRESS]
DEFAULT_ENVIRONMENT_NAME = 'production'

def backfill_monitor_environments(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Project = apps.get_model('sentry', 'Project')
    Monitor = apps.get_model('sentry', 'Monitor')
    Environment = apps.get_model('sentry', 'Environment')
    EnvironmentProject = apps.get_model('sentry', 'EnvironmentProject')
    MonitorEnvironment = apps.get_model('sentry', 'MonitorEnvironment')
    queryset = RangeQuerySetWrapperWithProgressBar(Monitor.objects.filter(monitorenvironment__isnull=True).exclude(status__in=[MonitorStatus.PENDING_DELETION, MonitorStatus.DELETION_IN_PROGRESS]).values_list('id', 'organization_id', 'project_id', 'status', 'next_checkin', 'last_checkin'), result_value_getter=lambda item: item[0])
    for (monitor_id, organization_id, project_id, status, next_checkin, last_checkin) in queryset:
        try:
            Project.objects.get(id=project_id)
        except Project.DoesNotExist:
            continue
        environment = Environment.objects.get_or_create(name=DEFAULT_ENVIRONMENT_NAME, organization_id=organization_id)[0]
        EnvironmentProject.objects.get_or_create(project_id=project_id, environment=environment, defaults={'is_hidden': None})
        monitorenvironment_defaults = {'status': status, 'next_checkin': next_checkin, 'last_checkin': last_checkin}
        MonitorEnvironment.objects.get_or_create(monitor_id=monitor_id, environment=environment, defaults=monitorenvironment_defaults)

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0379_create_notificationaction_model')]
    operations = [migrations.RunPython(backfill_monitor_environments, migrations.RunPython.noop, hints={'tables': ['sentry_project', 'sentry_monitor', 'sentry_monitorenvironment', 'sentry_environment', 'sentry_environmentproject']})]
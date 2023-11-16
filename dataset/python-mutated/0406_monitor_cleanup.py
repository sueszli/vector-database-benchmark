from django.db import migrations
from sentry.monitors.models import MonitorStatus
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def clean_up_monitors(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Monitor = apps.get_model('sentry', 'Monitor')
    Organization = apps.get_model('sentry', 'Organization')
    Project = apps.get_model('sentry', 'Project')
    queryset = RangeQuerySetWrapperWithProgressBar(Monitor.objects.filter(monitorenvironment__isnull=True).exclude(status__in=[MonitorStatus.PENDING_DELETION, MonitorStatus.DELETION_IN_PROGRESS]).values_list('id', 'organization_id', 'project_id'), result_value_getter=lambda item: item[0])
    monitors_to_clean_up = []
    for (monitor_id, organization_id, project_id) in queryset:
        try:
            Organization.objects.get(id=organization_id)
            Project.objects.get(id=project_id)
        except (Organization.DoesNotExist, Project.DoesNotExist):
            monitors_to_clean_up.append(monitor_id)
    Monitor.objects.filter(id__in=monitors_to_clean_up).delete()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0405_rulesnooze_user_null')]
    operations = [migrations.RunPython(clean_up_monitors, migrations.RunPython.noop, hints={'tables': ['sentry_monitor', 'sentry_organization', 'sentry_project']})]
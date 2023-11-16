from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration

def set_monitor_status_to_object_status_only(apps, schema_editor):
    if False:
        return 10
    Monitor = apps.get_model('sentry', 'Monitor')
    for monitor in Monitor.objects.exclude(status__in=(0, 1, 2, 3)):
        monitor.status = 0
        monitor.save(update_fields=['status'])

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0432_backfill_org_member_id_organizationmembermapping')]
    operations = [migrations.RunPython(set_monitor_status_to_object_status_only, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_monitor']})]
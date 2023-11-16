from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_next_checkin_latest(apps, schema_editor):
    if False:
        while True:
            i = 10
    MonitorEnvironment = apps.get_model('sentry', 'MonitorEnvironment')
    for monitor_environment in RangeQuerySetWrapperWithProgressBar(MonitorEnvironment.objects.all()):
        if monitor_environment.next_checkin_latest is not None or monitor_environment.next_checkin is None:
            continue
        monitor_environment.next_checkin_latest = monitor_environment.next_checkin
        monitor_environment.save(update_fields=['next_checkin_latest'])

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0526_pr_comment_type_column')]
    operations = [migrations.RunPython(backfill_next_checkin_latest, migrations.RunPython.noop, hints={'tables': ['sentry_monitorenvironment']})]
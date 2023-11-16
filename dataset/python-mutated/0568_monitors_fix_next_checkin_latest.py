from datetime import timedelta
from django.db import migrations
from django.db.models import F
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def fix_next_checkin_latest(apps, schema_editor):
    if False:
        return 10
    MonitorEnvironment = apps.get_model('sentry', 'MonitorEnvironment')
    query = MonitorEnvironment.objects.select_related('monitor').filter(next_checkin=F('next_checkin_latest'))
    for monitor_env in RangeQuerySetWrapperWithProgressBar(query):
        margin = monitor_env.monitor.config.get('checkin_margin') or 1
        monitor_env.next_checkin_latest = monitor_env.next_checkin + timedelta(minutes=margin)
        monitor_env.save()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0567_add_slug_reservation_model')]
    operations = [migrations.RunPython(fix_next_checkin_latest, migrations.RunPython.noop, hints={'tables': ['sentry_monitor']})]
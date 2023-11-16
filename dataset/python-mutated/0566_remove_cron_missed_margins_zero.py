from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def remove_missed_margins_zero(apps, schema_editor):
    if False:
        print('Hello World!')
    Monitor = apps.get_model('sentry', 'Monitor')
    for monitor in RangeQuerySetWrapperWithProgressBar(Monitor.objects.all()):
        margin = monitor.config.get('checkin_margin')
        if margin == 0:
            monitor.config['checkin_margin'] = None
            monitor.save()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0565_fix_diff_env_dupe_alerts')]
    operations = [migrations.RunPython(remove_missed_margins_zero, migrations.RunPython.noop, hints={'tables': ['sentry_monitor']})]
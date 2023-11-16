from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def migrate_monitor_slugs(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Monitor = apps.get_model('sentry', 'Monitor')
    for monitor in RangeQuerySetWrapperWithProgressBar(Monitor.objects.filter()):
        if monitor.slug is not None:
            continue
        monitor.slug = str(monitor.guid)
        monitor.save()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0366_add_slug_to_monitors')]
    operations = [migrations.RunPython(migrate_monitor_slugs, migrations.RunPython.noop, hints={'tables': ['sentry_monitior']})]
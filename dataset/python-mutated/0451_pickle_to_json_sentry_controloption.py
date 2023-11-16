from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def _backfill(apps, schema_editor):
    if False:
        print('Hello World!')
    ControlOption = apps.get_model('sentry', 'ControlOption')
    for obj in RangeQuerySetWrapperWithProgressBar(ControlOption.objects.all()):
        obj.save(update_fields=['value'])

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0450_pickle_to_json_sentry_option')]
    operations = [migrations.RunPython(_backfill, migrations.RunPython.noop, hints={'tables': ['sentry_controloption']})]
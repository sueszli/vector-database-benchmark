from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def _backfill(apps, schema_editor):
    if False:
        print('Hello World!')
    cls = apps.get_model('sentry', 'UserOption')
    for obj in RangeQuerySetWrapperWithProgressBar(cls.objects.all()):
        obj.save(update_fields=['value'])

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0454_pickle_to_json_sentry_projectoptions')]
    operations = [migrations.RunPython(_backfill, migrations.RunPython.noop, hints={'tables': ['sentry_useroption']})]
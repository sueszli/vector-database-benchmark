from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBarApprox

def _backfill(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    cls = apps.get_model('sentry', 'RawEvent')
    for obj in RangeQuerySetWrapperWithProgressBarApprox(cls.objects.all()):
        obj.save(update_fields=['data'])

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0467_control_files')]
    operations = [migrations.RunPython(_backfill, migrations.RunPython.noop, hints={'tables': ['sentry_rawevent']})]
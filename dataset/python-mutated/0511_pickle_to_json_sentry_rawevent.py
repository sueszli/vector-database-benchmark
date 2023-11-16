import logging
from django.db import migrations
from django.db.utils import DatabaseError
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBarApprox

def _backfill(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    cls = apps.get_model('sentry', 'RawEvent')
    for obj in RangeQuerySetWrapperWithProgressBarApprox(cls.objects.all()):
        try:
            obj.save(update_fields=['data'])
        except DatabaseError as e:
            logging.warning(f'ignoring save error (row was likely deleted): {e}')

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0510_index_checkin_traceid')]
    operations = [migrations.RunPython(_backfill, migrations.RunPython.noop, hints={'tables': ['sentry_rawevent']})]
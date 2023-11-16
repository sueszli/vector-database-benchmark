import logging
from django.db import migrations
from django.db.utils import DatabaseError
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBarApprox

def _backfill(apps, schema_editor):
    if False:
        return 10
    cls = apps.get_model('sentry', 'Activity')
    for obj in RangeQuerySetWrapperWithProgressBarApprox(cls.objects.all()):
        try:
            obj.save(update_fields=['data'])
        except DatabaseError as e:
            logging.warning(f'ignoring save error (row was likely deleted): {e}')

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0492_pickle_to_json_sentry_groupedmessage')]
    operations = [migrations.RunPython(_backfill, migrations.RunPython.noop, hints={'tables': ['sentry_activity']})]
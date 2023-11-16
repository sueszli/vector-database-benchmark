import logging
from django.db import migrations
from django.db.utils import DatabaseError
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBarApprox

def _backfill(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    cls = apps.get_model('sentry', 'Group')
    for obj in RangeQuerySetWrapperWithProgressBarApprox(cls.objects.all()):
        try:
            obj.save(update_fields=['data'])
        except DatabaseError as e:
            logging.warning(f'ignoring save error (row was likely deleted): {e}')

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0491_remove_orgmemmap_unique_constraints')]
    operations = [migrations.RunPython(_backfill, migrations.RunPython.noop, hints={'tables': ['sentry_groupedmessage']})]
from django.db import migrations
from django.db.utils import IntegrityError
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_use_case_id(apps, schema_editor):
    if False:
        while True:
            i = 10
    '\n    When the `use_case_id` was added to the sentry_perfstringindexer table\n    the default was "performance". We now want the use_case_id to align\n    with the relay namespace, which should be "transactions".\n\n    This migration may run after we\'ve starting to write new rows that\n    have the correct "transactions" use_case_id, so we just want to fix the\n    rows that still have the "performance" use_case_id.\n\n    Ultimately the default for this column should be removed, as we add\n    more use cases.\n\n    '
    PerfStringIndexer = apps.get_model('sentry', 'PerfStringIndexer')
    for indexed_str in RangeQuerySetWrapperWithProgressBar(PerfStringIndexer.objects.all()):
        if indexed_str.use_case_id == 'performance':
            indexed_str.use_case_id = 'transactions'
            try:
                indexed_str.save(update_fields=['use_case_id'])
            except IntegrityError:
                pass

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0445_drop_deprecated_monitor_next_last_checkin_db_op')]
    operations = [migrations.RunPython(backfill_use_case_id, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_perfstringindexer']})]
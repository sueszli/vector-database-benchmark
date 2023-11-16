from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def _backfill(apps, schema_editor):
    if False:
        while True:
            i = 10
    cls = apps.get_model('sentry', 'ProcessingIssue')
    for obj in RangeQuerySetWrapperWithProgressBar(cls.objects.all()):
        obj.save(update_fields=['data'])

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0462_pickle_to_json_sentry_grouptombstone')]
    operations = [migrations.RunPython(_backfill, migrations.RunPython.noop, hints={'tables': ['sentry_processingissue']})]
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBarApprox

def _backfill(apps, schema_editor):
    if False:
        print('Hello World!')
    cls = apps.get_model('sentry', 'Group')
    for obj in RangeQuerySetWrapperWithProgressBarApprox(cls.objects.all()):
        obj.save(update_fields=['data'])

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0460_pickle_to_json_sentry_auditlogentry')]
    operations = [migrations.RunPython(_backfill, migrations.RunPython.noop, hints={'tables': ['sentry_groupedmessage']})]
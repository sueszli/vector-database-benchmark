from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def _backfill(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Option = apps.get_model('sentry', 'Option')
    for obj in RangeQuerySetWrapperWithProgressBar(Option.objects.all()):
        obj.save(update_fields=['value'])

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0449_pickle_to_json_authenticator')]
    operations = [migrations.RunPython(_backfill, migrations.RunPython.noop, hints={'tables': ['sentry_option']})]
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def _backfill_authenticator(apps, schema_editor):
    if False:
        print('Hello World!')
    Authenticator = apps.get_model('sentry', 'Authenticator')
    for obj in RangeQuerySetWrapperWithProgressBar(Authenticator.objects.all()):
        obj.save(update_fields=['config'])

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0448_add_expected_time_config_to_cron_checkin')]
    operations = [migrations.RunPython(_backfill_authenticator, migrations.RunPython.noop, hints={'tables': ['sentry_authenticator']})]
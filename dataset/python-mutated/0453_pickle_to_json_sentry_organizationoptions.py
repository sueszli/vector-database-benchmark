from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def _backfill(apps, schema_editor):
    if False:
        return 10
    cls = apps.get_model('sentry', 'OrganizationOption')
    for obj in RangeQuerySetWrapperWithProgressBar(cls.objects.all()):
        obj.save(update_fields=['value'])

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0452_break_user_actor_fk')]
    operations = [migrations.RunPython(_backfill, migrations.RunPython.noop, hints={'tables': ['sentry_organizationoptions']})]
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_app_id_str(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    AppConnectBuild = apps.get_model('sentry', 'AppConnectBuild')
    for appconnect_build in RangeQuerySetWrapperWithProgressBar(AppConnectBuild.objects.all()):
        appconnect_build.app_id_str = str(appconnect_build.app_id)
        appconnect_build.save(update_fields=['app_id_str'])

class Migration(CheckedMigration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0327_add_appid_str')]
    operations = [migrations.RunPython(backfill_app_id_str, migrations.RunPython.noop, hints={'tables': ['sentry_appconnectbuild']})]
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_installation_ids(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    ServiceHook = apps.get_model('sentry', 'ServiceHook')
    queryset = RangeQuerySetWrapperWithProgressBar(ServiceHook.objects.filter(project_id__isnull=True))
    for hook in queryset:
        hook.installation_id = hook.actor_id
        hook.save()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0383_mv_user_avatar')]
    operations = [migrations.RunPython(backfill_installation_ids, migrations.RunPython.noop, hints={'tables': ['sentry_servicehook']})]
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_msteams_installation_type(apps, schema_editor):
    if False:
        return 10
    Integration = apps.get_model('sentry', 'Integration')
    for integration in RangeQuerySetWrapperWithProgressBar(Integration.objects.filter(provider='msteams')):
        if integration.metadata and integration.metadata.get('installation_type'):
            continue
        integration.metadata.update({'installation_type': 'team'})
        integration.save()

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0302_mep_backfill_and_not_null_snuba_query_type')]
    operations = [migrations.RunPython(backfill_msteams_installation_type, migrations.RunPython.noop, hints={'tables': ['sentry_integration']})]
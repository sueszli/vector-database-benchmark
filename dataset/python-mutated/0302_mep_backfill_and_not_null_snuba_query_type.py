from django.db import migrations, models
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar
dataset_to_type_map = {'events': 0, 'transactions': 1, 'sessions': 2, 'metrics': 2}

def backfill_snubaquery_type(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    SnubaQuery = apps.get_model('sentry', 'SnubaQuery')
    for snuba_query in RangeQuerySetWrapperWithProgressBar(SnubaQuery.objects.all()):
        if snuba_query.type is not None:
            continue
        snuba_query.type = dataset_to_type_map[snuba_query.dataset]
        snuba_query.save(update_fields=['type'])

class Migration(CheckedMigration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0301_mep_move_type_to_snuba_query')]
    operations = [migrations.RunPython(backfill_snubaquery_type, migrations.RunPython.noop, hints={'tables': ['sentry_snubaquery']}), migrations.AlterField(model_name='snubaquery', name='type', field=models.SmallIntegerField())]
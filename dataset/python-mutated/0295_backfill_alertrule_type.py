from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar
dataset_to_type_map = {'events': 0, 'transactions': 1, 'sessions': 2, 'metrics': 2}

def backfill_alertrule_type(apps, schema_editor):
    if False:
        print('Hello World!')
    AlertRule = apps.get_model('sentry', 'AlertRule')
    for alert_rule in RangeQuerySetWrapperWithProgressBar(AlertRule.objects_with_snapshots.select_related('snuba_query').all()):
        if alert_rule.type is not None or alert_rule.snuba_query is None:
            continue
        alert_rule.type = dataset_to_type_map[alert_rule.snuba_query.dataset]
        alert_rule.save(update_fields=['type'])

class Migration(CheckedMigration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0294_alertrule_type')]
    operations = [migrations.RunPython(backfill_alertrule_type, migrations.RunPython.noop, hints={'tables': ['sentry_alertrule']})]
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def remove_name_data(apps, schema_editor):
    if False:
        print('Hello World!')
    Rule = apps.get_model('sentry', 'Rule')
    for rule in RangeQuerySetWrapperWithProgressBar(Rule.objects.all()):
        for action in rule.data.get('actions', []):
            if action.get('name') or action.get('name') in [0, '']:
                del action['name']
        for condition in rule.data.get('conditions', []):
            if condition.get('name') or condition.get('name') in [0, '']:
                del condition['name']
        rule.save(update_fields=['data'])

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0537_backfill_xactor_team_and_user_ids')]
    operations = [migrations.RunPython(remove_name_data, migrations.RunPython.noop, hints={'tables': ['sentry_rule']})]
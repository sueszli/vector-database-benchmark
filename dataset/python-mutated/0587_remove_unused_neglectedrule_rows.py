from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def remove_unused_neglectedrule_rows(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    NeglectedRule = apps.get_model('sentry', 'NeglectedRule')
    for neglected_rule in RangeQuerySetWrapperWithProgressBar(NeglectedRule.objects.all()):
        if neglected_rule.sent_initial_email_date and (not neglected_rule.sent_final_email_date):
            neglected_rule.delete()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0586_add_has_feedbacks_flag')]
    operations = [migrations.RunPython(remove_unused_neglectedrule_rows, migrations.RunPython.noop, hints={'tables': ['sentry_neglectedrule']})]
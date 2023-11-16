from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_alertrule_organization(apps, schema_editor):
    if False:
        while True:
            i = 10
    AlertRule = apps.get_model('sentry', 'AlertRule')
    Organization = apps.get_model('sentry', 'Organization')
    for alert_rule in RangeQuerySetWrapperWithProgressBar(AlertRule.objects_with_snapshots.all()):
        if not alert_rule.organization_id:
            continue
        try:
            correct_org = Organization.objects.filter(project__querysubscription__snuba_query__alertrule__id=alert_rule.id).distinct().get()
        except Organization.DoesNotExist:
            continue
        if alert_rule.organization_id != correct_org.id:
            alert_rule.organization_id = correct_org.id
            alert_rule.save()

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0285_add_organization_member_team_role')]
    operations = [migrations.RunPython(backfill_alertrule_organization, migrations.RunPython.noop, hints={'tables': ['sentry_alertrule']})]
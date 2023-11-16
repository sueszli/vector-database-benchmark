from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_alert_owners(apps, schema_editor):
    if False:
        while True:
            i = 10
    AlertRule = apps.get_model('sentry', 'AlertRule')
    OrganizationMember = apps.get_model('sentry', 'OrganizationMember')
    User = apps.get_model('sentry', 'User')
    Team = apps.get_model('sentry', 'Team')
    for alert_rule in RangeQuerySetWrapperWithProgressBar(AlertRule.objects_with_snapshots.select_related('owner').all()):
        owner = alert_rule.owner
        if not owner:
            continue
        valid_owner = False
        if owner.type == 1:
            user = User.objects.get(actor_id=owner.id)
            if OrganizationMember.objects.filter(organization_id=alert_rule.organization_id, user_id=user.id).exists():
                valid_owner = True
        elif Team.objects.filter(actor_id=owner.id, organization_id=alert_rule.organization_id).exists():
            valid_owner = True
        if not valid_owner:
            alert_rule.owner = None
            alert_rule.save()

class Migration(CheckedMigration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0296_alertrule_type_not_null')]
    operations = [migrations.RunPython(backfill_alert_owners, migrations.RunPython.noop, hints={'tables': ['sentry_alertrule']})]
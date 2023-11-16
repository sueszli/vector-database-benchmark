from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_om_is_active(apps, schema_editor):
    if False:
        while True:
            i = 10
    OrganizationMember = apps.get_model('sentry', 'OrganizationMember')
    for om in RangeQuerySetWrapperWithProgressBar(OrganizationMember.objects.all().select_related('user')):
        if om.user and (not om.user.is_active):
            om.user_is_active = False
            om.save(update_fields=['user_is_active'])

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0472_delete_past_organization_mappings')]
    operations = [migrations.RunPython(backfill_om_is_active, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_organizationmember']})]
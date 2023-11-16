from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_om_user_email(apps, schema_editor):
    if False:
        return 10
    OrganizationMember = apps.get_model('sentry', 'OrganizationMember')
    for om in RangeQuerySetWrapperWithProgressBar(OrganizationMember.objects.all().select_related('user')):
        if om.user and om.user_email is None:
            om.user_email = om.user.email
            om.save(update_fields=['user_email'])

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0482_denormalize_user_email')]
    operations = [migrations.RunPython(backfill_om_user_email, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_organizationmember']})]
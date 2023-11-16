from django.db import IntegrityError, migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_org_membermapping(apps, schema_editor):
    if False:
        while True:
            i = 10
    OrganizationMember = apps.get_model('sentry', 'OrganizationMember')
    OrganizationMemberMapping = apps.get_model('sentry', 'OrganizationMemberMapping')
    OrganizationMemberMapping.objects.filter(organizationmember_id__isnull=True).delete()
    for member in RangeQuerySetWrapperWithProgressBar(OrganizationMember.objects.all()):
        mapping = OrganizationMemberMapping.objects.filter(organization_id=member.organization_id, organizationmember_id=member.id).first()
        if mapping:
            continue
        try:
            OrganizationMemberMapping.objects.create(organization_id=member.organization_id, organizationmember_id=member.id, role=member.role, user_id=member.user_id, email=member.email, inviter_id=member.inviter_id, invite_status=member.invite_status)
        except IntegrityError:
            pass

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0468_pickle_to_json_sentry_rawevent')]
    operations = [migrations.RunPython(backfill_org_membermapping, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_organizationmembermapping', 'sentry_organizationmember']})]
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_org_member_id_organizationmembermapping(apps, schema_editor):
    if False:
        while True:
            i = 10
    OrganizationMemberMapping = apps.get_model('sentry', 'OrganizationMemberMapping')
    OrganizationMember = apps.get_model('sentry', 'OrganizationMember')
    for org_member_mapping in RangeQuerySetWrapperWithProgressBar(OrganizationMemberMapping.objects.filter(organizationmember_id__isnull=True)):
        org_member = None
        try:
            org_member = OrganizationMember.objects.filter(organization_id=org_member_mapping.organization_id, user_id=org_member_mapping.user_id, email=org_member_mapping.email).get()
        except OrganizationMember.DoesNotExist:
            org_member = None
        if org_member is not None:
            org_member_mapping.organizationmember_id = org_member.id
            org_member_mapping.save(update_fields=['organizationmember_id'])

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0431_breaking_orgintegration_pieces_and_default_auth_team_fks')]
    operations = [migrations.RunPython(backfill_org_member_id_organizationmembermapping, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_organizationmembermapping', 'sentry_organizationmember']})]
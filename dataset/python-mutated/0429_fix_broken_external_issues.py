from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration

def fix_broken_external_issues(apps, schema_editor):
    if False:
        while True:
            i = 10
    ExternalIssue = apps.get_model('sentry', 'ExternalIssue')
    broken_ids = [636683, 636687, 636692]
    old_organization_id = 443715
    new_organization_id = 5417824
    for issue in ExternalIssue.objects.filter(id__in=broken_ids, organization_id=old_organization_id):
        issue.organization_id = new_organization_id
        issue.save()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0428_backfill_denormalize_notification_actor')]
    operations = [migrations.RunPython(fix_broken_external_issues, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_externalissue']})]
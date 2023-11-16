from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_project_ownership(apps, schema_editor):
    if False:
        return 10
    ProjectOwnership = apps.get_model('sentry', 'ProjectOwnership')
    ProjectCodeOwners = apps.get_model('sentry', 'ProjectCodeOwners')
    for codeowner in RangeQuerySetWrapperWithProgressBar(ProjectCodeOwners.objects.all()):
        ProjectOwnership.objects.get_or_create(project_id=codeowner.project_id, defaults={'auto_assignment': False, 'suspect_committer_auto_assignment': False})

class Migration(CheckedMigration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0322_projectownership_fallthrough_revert')]
    operations = [migrations.RunPython(backfill_project_ownership, migrations.RunPython.noop, hints={'tables': ['sentry_projectownership', 'sentry_projectcodeowners']})]
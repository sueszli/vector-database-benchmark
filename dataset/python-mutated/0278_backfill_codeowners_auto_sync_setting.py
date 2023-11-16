from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_codeowners_auto_sync_column(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    ProjectOwnership = apps.get_model('sentry', 'ProjectOwnership')
    for ownership in RangeQuerySetWrapperWithProgressBar(ProjectOwnership.objects.all()):
        if ownership.codeowners_auto_sync is None:
            ownership.codeowners_auto_sync = True
        ownership.save()

class Migration(CheckedMigration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0277_backfill_dashboard_widget_query_columns_aggregates')]
    operations = [migrations.RunPython(backfill_codeowners_auto_sync_column, migrations.RunPython.noop, hints={'tables': ['sentry_projectownership']})]
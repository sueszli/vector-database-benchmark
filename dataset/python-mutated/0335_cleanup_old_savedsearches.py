from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def cleanup_savedsearch(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    SavedSearch = apps.get_model('sentry', 'SavedSearch')
    for search in RangeQuerySetWrapperWithProgressBar(SavedSearch.objects.all()):
        if search.project_id is not None and search.organization_id is None:
            search.delete()

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0334_repositorypath_automatically_generated')]
    operations = [migrations.RunPython(cleanup_savedsearch, migrations.RunPython.noop, hints={'tables': ['sentry_savedsearch']})]
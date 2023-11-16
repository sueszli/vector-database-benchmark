from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def cleanup_pinned_searches(apps, schema_editor):
    if False:
        return 10
    SavedSearch = apps.get_model('sentry', 'SavedSearch')
    for search in RangeQuerySetWrapperWithProgressBar(SavedSearch.objects.all()):
        if search.owner is not None:
            search.visibility = 'owner_pinned'
        else:
            search.visibility = 'organization'
        search.save()

class Migration(CheckedMigration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0338_add_saved_search_visibility')]
    operations = [migrations.RunPython(cleanup_pinned_searches, migrations.RunPython.noop, hints={'tables': ['sentry_savedsearch']})]
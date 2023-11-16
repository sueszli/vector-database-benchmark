from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def migrate_saved_searches(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    SavedSearch = apps.get_model('sentry', 'SavedSearch')
    for search in RangeQuerySetWrapperWithProgressBar(SavedSearch.objects.all()):
        if search.sort == 'betterPriority':
            search.sort = 'priority'
            search.save()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0513_django_jsonfield')]
    operations = [migrations.RunPython(migrate_saved_searches, migrations.RunPython.noop, hints={'tables': ['sentry_savedsearch']})]
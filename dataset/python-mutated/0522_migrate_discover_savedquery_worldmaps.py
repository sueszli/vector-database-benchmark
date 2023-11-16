from django.db import migrations
from sentry.discover.models import DiscoverSavedQuery
from sentry.new_migrations.migrations import CheckedMigration

def migrate_savedquery_worldmap_display_to_totalPeriod(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    savedQueries = DiscoverSavedQuery.objects.filter(query__contains={'display': 'worldmap'})
    for savedQuery in savedQueries:
        savedQuery.query['display'] = 'default'
        savedQuery.save()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0521_migrate_world_map_widgets')]
    operations = [migrations.RunPython(migrate_savedquery_worldmap_display_to_totalPeriod, migrations.RunPython.noop, hints={'tables': ['sentry_discoversavedquery']})]
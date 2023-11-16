from typing import List
from posthog.async_migrations.definition import AsyncMigrationDefinition, AsyncMigrationOperation
'\nNooping this migration for future compatibility. Superseded by 0007_persons_and_groups_on_events_backfill.\n\nIf users ran the old version of this, they will be ok to run 0007, if not, they will also be ok to run it.\n'

class Migration(AsyncMigrationDefinition):
    description = 'No-op migration'
    posthog_max_version = '1.41.99'
    depends_on = '0005_person_replacing_by_version'
    operations: List[AsyncMigrationOperation] = []

    def is_required(self):
        if False:
            for i in range(10):
                print('nop')
        return False
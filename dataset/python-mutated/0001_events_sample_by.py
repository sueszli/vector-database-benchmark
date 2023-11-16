from typing import List
from posthog.async_migrations.definition import AsyncMigrationDefinition, AsyncMigrationOperation
'\nNooping this migration for future compatibility. Superseded by 0002_events_sample_by.\n\nIf users ran the old version of this, they will be ok to run 0002, if not, they will also be ok to run it.\n'

class Migration(AsyncMigrationDefinition):
    description = 'Test migration'
    posthog_max_version = '1.33.9'
    operations: List[AsyncMigrationOperation] = []

    def is_required(self):
        if False:
            i = 10
            return i + 15
        return False
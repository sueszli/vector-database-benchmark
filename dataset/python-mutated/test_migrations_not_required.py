import pytest
from posthog.async_migrations.setup import ALL_ASYNC_MIGRATIONS
from posthog.async_migrations.test.util import AsyncMigrationBaseTest
from posthog.client import sync_execute
from posthog.models.person.sql import COMMENT_DISTINCT_ID_COLUMN_SQL
pytestmark = pytest.mark.async_migrations

class TestAsyncMigrationsNotRequired(AsyncMigrationBaseTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        sync_execute(COMMENT_DISTINCT_ID_COLUMN_SQL())
        sync_execute('TRUNCATE TABLE sharded_events')

    def test_async_migrations_not_required_on_fresh_instances(self):
        if False:
            return 10
        for (name, migration) in ALL_ASYNC_MIGRATIONS.items():
            self.assertFalse(migration.is_required(), f'Async migration {name} is_required returned True')
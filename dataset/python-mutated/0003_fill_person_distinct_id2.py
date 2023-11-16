from functools import cached_property
from posthog.async_migrations.definition import AsyncMigrationDefinition, AsyncMigrationOperationSQL
from posthog.client import sync_execute
from posthog.constants import AnalyticsDBMS
from posthog.settings import CLICKHOUSE_DATABASE
'\nMigration summary:\n\nSchema change to migrate the data from the old person_distinct_id table\nto the new person_distinct_id2 table.\n\nThe reason this is needed is for faster `person_distinct_id` queries as the\nold schema worked off of (distinct_id, person_id) pairs, making it expensive\nto for our analytics queries, which need to map from distinct_id -> latest person_id.\n\nThe new schema works off of distinct_id columns, leveraging ReplacingMergeTrees\nwith a version column we store in postgres.\n\nWe migrate teams one-by-one to avoid running out of memory.\n\nThe migration strategy:\n\n    1. write to both pdi and pdi2 any new updates (done prior to this migration)\n    2. insert all non-deleted (team_id, distinct_id, person_id) rows from pdi into pdi2 (this migration)\n    3. Once migration has run, we only read/write from/to pdi2.\n'

class Migration(AsyncMigrationDefinition):
    description = 'Set up person_distinct_id2 table, speeding up person-related queries.'
    depends_on = '0002_events_sample_by'
    posthog_min_version = '1.33.0'
    posthog_max_version = '1.33.9'

    def is_required(self):
        if False:
            return 10
        rows = sync_execute('\n            SELECT comment\n            FROM system.columns\n            WHERE database = %(database)s\n        ', {'database': CLICKHOUSE_DATABASE})
        comments = [row[0] for row in rows]
        return 'skip_0003_fill_person_distinct_id2' not in comments

    @cached_property
    def operations(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.migrate_team_operation(team_id) for team_id in self._team_ids]

    def migrate_team_operation(self, team_id: int):
        if False:
            print('Hello World!')
        return AsyncMigrationOperationSQL(database=AnalyticsDBMS.CLICKHOUSE, sql=f'\n                INSERT INTO person_distinct_id2(team_id, distinct_id, person_id, is_deleted, version)\n                SELECT\n                    team_id,\n                    distinct_id,\n                    argMax(person_id, _timestamp) as person_id,\n                    0 as is_deleted,\n                    0 as version\n                FROM (\n                    SELECT\n                        distinct_id,\n                        person_id,\n                        any(team_id) as team_id,\n                        max(_timestamp) as _timestamp\n                    FROM\n                        person_distinct_id\n                    WHERE\n                        person_distinct_id.team_id = {team_id}\n                    GROUP BY\n                        person_id,\n                        distinct_id\n                    HAVING\n                        max(is_deleted) = 0\n                )\n                GROUP BY team_id, distinct_id\n            ', rollback=None)

    @cached_property
    def _team_ids(self):
        if False:
            for i in range(10):
                print('nop')
        return list(sorted((row[0] for row in sync_execute('SELECT DISTINCT team_id FROM person_distinct_id'))))
from typing import Any, Dict, List, Set, Tuple
from posthog.client import sync_execute
from posthog.models.async_deletion import AsyncDeletion, DeletionType
from posthog.models.async_deletion.delete import AsyncDeletionProcess, logger
from posthog.settings.data_stores import CLICKHOUSE_CLUSTER
TABLES_TO_DELETE_TEAM_DATA_FROM = ['person', 'person_distinct_id', 'person_distinct_id2', 'groups', 'cohortpeople', 'person_static_cohort', 'plugin_log_entries']

class AsyncEventDeletion(AsyncDeletionProcess):
    DELETION_TYPES = [DeletionType.Team, DeletionType.Group, DeletionType.Person]

    def process(self, deletions: List[AsyncDeletion]):
        if False:
            return 10
        if len(deletions) == 0:
            logger.debug('No AsyncDeletion to perform')
            return
        logger.info('Starting AsyncDeletion on `events` table in ClickHouse', {'count': len(deletions), 'team_ids': list(set((row.team_id for row in deletions)))})
        (conditions, args) = self._conditions(deletions)
        sync_execute(f"\n            DELETE FROM sharded_events\n            ON CLUSTER '{CLICKHOUSE_CLUSTER}'\n            WHERE {' OR '.join(conditions)}\n            ", args)
        team_deletions = [row for row in deletions if row.deletion_type == DeletionType.Team]
        if len(team_deletions) == 0:
            return
        logger.info('Starting AsyncDeletion for teams on other tables', {'count': len(team_deletions), 'team_ids': list(set((row.team_id for row in deletions)))})
        (conditions, args) = self._conditions(team_deletions)
        for table in TABLES_TO_DELETE_TEAM_DATA_FROM:
            sync_execute(f"\n                DELETE FROM {table}\n                ON CLUSTER '{CLICKHOUSE_CLUSTER}'\n                WHERE {' OR '.join(conditions)}\n                ", args)

    def _verify_by_group(self, deletion_type: int, async_deletions: List[AsyncDeletion]) -> List[AsyncDeletion]:
        if False:
            while True:
                i = 10
        if deletion_type == DeletionType.Team:
            team_ids_with_data = self._verify_by_column('team_id', async_deletions)
            return [row for row in async_deletions if (row.team_id,) not in team_ids_with_data]
        elif deletion_type in (DeletionType.Person, DeletionType.Group):
            columns = f'team_id, {self._column_name(async_deletions[0])}'
            with_data = set(((team_id, str(key)) for (team_id, key) in self._verify_by_column(columns, async_deletions)))
            return [row for row in async_deletions if (row.team_id, row.key) not in with_data]
        else:
            return []

    def _verify_by_column(self, distinct_columns: str, async_deletions: List[AsyncDeletion]) -> Set[Tuple[Any, ...]]:
        if False:
            i = 10
            return i + 15
        (conditions, args) = self._conditions(async_deletions)
        clickhouse_result = sync_execute(f"\n            SELECT DISTINCT {distinct_columns}\n            FROM events\n            WHERE {' OR '.join(conditions)}\n            ", args)
        return set((tuple(row) for row in clickhouse_result))

    def _column_name(self, async_deletion: AsyncDeletion):
        if False:
            for i in range(10):
                print('nop')
        assert async_deletion.deletion_type in (DeletionType.Person, DeletionType.Group)
        if async_deletion.deletion_type == DeletionType.Person:
            return 'person_id'
        else:
            return f'$group_{async_deletion.group_type_index}'

    def _condition(self, async_deletion: AsyncDeletion, suffix: str) -> Tuple[str, Dict]:
        if False:
            print('Hello World!')
        if async_deletion.deletion_type == DeletionType.Team:
            return (f'team_id = %(team_id{suffix})s', {f'team_id{suffix}': async_deletion.team_id})
        else:
            return (f'(team_id = %(team_id{suffix})s AND {self._column_name(async_deletion)} = %(key{suffix})s)', {f'team_id{suffix}': async_deletion.team_id, f'key{suffix}': async_deletion.key})
from time import sleep
from uuid import UUID, uuid4
import pytest
from django.conf import settings
from posthog.client import sync_execute
from posthog.conftest import create_clickhouse_tables
from posthog.management.commands.backfill_persons_and_groups_on_events import run_backfill
from posthog.models.event.sql import EVENTS_DATA_TABLE
from posthog.test.base import BaseTest, ClickhouseTestMixin

def create_test_events(properties=''):
    if False:
        for i in range(10):
            print('nop')
    sync_execute(f"\n        INSERT INTO {EVENTS_DATA_TABLE()} (event, team_id, uuid, timestamp, distinct_id, properties)\n        VALUES\n            ('event1', 1, '{str(uuid4())}', now(), 'some_distinct_id', '{properties}')\n            ('event2', 1, '{str(uuid4())}', now(), 'some_distinct_id', '{properties}')\n        ")

@pytest.mark.ee
class TestBackfillPersonsAndGroupsOnEvents(BaseTest, ClickhouseTestMixin):

    def tearDown(self):
        if False:
            print('Hello World!')
        self.recreate_database()
        super().tearDown()

    def recreate_database(self, create_tables=True):
        if False:
            i = 10
            return i + 15
        sync_execute(f'DROP DATABASE {settings.CLICKHOUSE_DATABASE} SYNC')
        sync_execute(f'CREATE DATABASE {settings.CLICKHOUSE_DATABASE}')
        if create_tables:
            create_clickhouse_tables(0)

    def test_person_backfill(self):
        if False:
            return 10
        self.recreate_database()
        create_test_events()
        person_id = uuid4()
        person_props = '{ "foo": "bar" }'
        sync_execute(f"\n            INSERT INTO person (id, team_id, properties)\n            VALUES\n                ('{str(person_id)}', 1, '{person_props}')\n            ")
        sync_execute(f"\n            INSERT INTO person_distinct_id2 (person_id, distinct_id, team_id)\n            VALUES\n                ('{str(person_id)}', 'some_distinct_id', 1)\n            ")
        events_before = sync_execute('select event, person_id, person_properties from events')
        self.assertEqual(events_before, [('event1', UUID('00000000-0000-0000-0000-000000000000'), ''), ('event2', UUID('00000000-0000-0000-0000-000000000000'), '')])
        run_backfill({'team_id': 1, 'live_run': True})
        sleep(10)
        events_after = sync_execute('select event, person_id, person_properties from events')
        self.assertEqual(events_after, [('event1', person_id, '{ "foo": "bar" }'), ('event2', person_id, '{ "foo": "bar" }')])

    def test_groups_backfill(self):
        if False:
            for i in range(10):
                print('nop')
        self.recreate_database()
        create_test_events('{ "$group_0": "my_group" }')
        group_props = '{ "foo": "bar" }'
        sync_execute(f"\n            INSERT INTO groups (group_type_index, group_key, group_properties)\n            VALUES\n                (0, 'my_group', '{group_props}')\n            ")
        events_before = sync_execute('select event, $group_0, group0_properties from events')
        self.assertEqual(events_before, [('event1', 'my_group', ''), ('event2', 'my_group', '')])
        run_backfill({'team_id': 1, 'live_run': True})
        sleep(10)
        events_after = sync_execute('select event, $group_0, group0_properties from events')
        self.assertEqual(events_after, [('event1', 'my_group', group_props), ('event2', 'my_group', group_props)])
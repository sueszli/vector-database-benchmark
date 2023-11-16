import tap_tester.connections as connections
import tap_tester.runner as runner
from base import ZendeskTest
from tap_tester import menagerie
from datetime import datetime
import uuid
import os
import time

class ZendeskBookMark(ZendeskTest):
    """Test tap sets a bookmark and respects it for the next sync of a stream"""

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'zendesk_bookmark_test'

    def test_run(self):
        if False:
            while True:
                i = 10
        "\n        Verify that for each stream you can do a sync which records bookmarks.\n        That the bookmark is the maximum value sent to the target for the replication key.\n        That a second sync respects the bookmark\n            All data of the second sync is >= the bookmark from the first sync\n            The number of records in the 2nd sync is less then the first (This assumes that\n                new data added to the stream is done at a rate slow enough that you haven't\n                doubled the amount of data from the start date to the first sync between\n                the first sync and second sync run in this test)\n\n        Verify that for full table stream, all data replicated in sync 1 is replicated again in sync 2.\n\n        PREREQUISITE\n        For EACH stream that is incrementally replicated there are multiple rows of data with\n            different values for the replication key\n        "
        expected_streams = self.expected_check_streams()
        expected_replication_keys = self.expected_replication_keys()
        expected_replication_methods = self.expected_replication_method()
        conn_id = connections.ensure_connection(self)
        found_catalogs = self.run_and_verify_check_mode(conn_id)
        catalog_entries = [catalog for catalog in found_catalogs if catalog.get('tap_stream_id') in expected_streams]
        self.perform_and_verify_table_and_field_selection(conn_id, catalog_entries)
        first_sync_record_count = self.run_and_verify_sync(conn_id)
        first_sync_records = runner.get_records_from_target_output()
        first_sync_bookmarks = menagerie.get_state(conn_id)
        new_states = {'bookmarks': dict()}
        simulated_states = self.calculated_states_by_stream(first_sync_bookmarks)
        for (stream, new_state) in simulated_states.items():
            new_states['bookmarks'][stream] = new_state
        menagerie.set_state(conn_id, new_states)
        second_sync_record_count = self.run_and_verify_sync(conn_id)
        second_sync_records = runner.get_records_from_target_output()
        second_sync_bookmarks = menagerie.get_state(conn_id)
        for stream in expected_streams:
            with self.subTest(stream=stream):
                expected_replication_method = expected_replication_methods[stream]
                first_sync_count = first_sync_record_count.get(stream, 0)
                second_sync_count = second_sync_record_count.get(stream, 0)
                first_sync_messages = [record.get('data') for record in first_sync_records.get(stream, {}).get('messages', []) if record.get('action') == 'upsert']
                second_sync_messages = [record.get('data') for record in second_sync_records.get(stream, {}).get('messages', []) if record.get('action') == 'upsert']
                first_bookmark_key_value = first_sync_bookmarks.get('bookmarks', {stream: None}).get(stream)
                second_bookmark_key_value = second_sync_bookmarks.get('bookmarks', {stream: None}).get(stream)
                if expected_replication_method == self.INCREMENTAL:
                    replication_key = next(iter(expected_replication_keys[stream]))
                    first_bookmark_value = first_bookmark_key_value.get(replication_key)
                    second_bookmark_value = second_bookmark_key_value.get(replication_key)
                    first_bookmark_value_utc = self.convert_state_to_utc(first_bookmark_value)
                    second_bookmark_value_utc = self.convert_state_to_utc(second_bookmark_value)
                    simulated_bookmark_value = self.convert_state_to_utc(new_states['bookmarks'][stream][replication_key])
                    self.assertIsNotNone(first_bookmark_key_value)
                    self.assertIsNotNone(first_bookmark_value)
                    self.assertIsNotNone(second_bookmark_key_value)
                    self.assertIsNotNone(second_bookmark_value)
                    if not stream == 'users':
                        self.assertEqual(second_bookmark_value, first_bookmark_value)
                    else:
                        self.assertGreaterEqual(second_bookmark_value, first_bookmark_value)
                    for record in first_sync_messages:
                        replication_key_value = record.get(replication_key)
                        if stream == 'tickets':
                            replication_key_value = datetime.utcfromtimestamp(replication_key_value).strftime('%Y-%m-%dT%H:%M:%SZ')
                        self.assertLessEqual(replication_key_value, first_bookmark_value_utc, msg='First sync bookmark was set incorrectly, a record with a greater replication-key value was synced.')
                    for record in second_sync_messages:
                        replication_key_value = record.get(replication_key)
                        if stream == 'tickets':
                            replication_key_value = datetime.utcfromtimestamp(replication_key_value).strftime('%Y-%m-%dT%H:%M:%SZ')
                        self.assertGreaterEqual(replication_key_value, simulated_bookmark_value, msg='Second sync records do not repect the previous bookmark.')
                        self.assertLessEqual(replication_key_value, second_bookmark_value_utc, msg='Second sync bookmark was set incorrectly, a record with a greater replication-key value was synced.')
                elif expected_replication_method == self.FULL_TABLE:
                    self.assertIsNone(first_bookmark_key_value)
                    self.assertIsNone(second_bookmark_key_value)
                    if not stream in ['ticket_comments', 'ticket_audits', 'ticket_metrics']:
                        self.assertEqual(second_sync_count, first_sync_count)
                else:
                    raise NotImplementedError('INVALID EXPECTATIONS\t\tSTREAM: {} REPLICATION_METHOD: {}'.format(stream, expected_replication_method))
                if stream == 'tags' and second_sync_count == 0 and (first_sync_count == 0):
                    print(f"FULL_TABLE stream 'tags' replicated 0 records, stream not fully tested")
                    continue
                self.assertGreater(second_sync_count, 0, msg='We are not fully testing bookmarking for {}'.format(stream))
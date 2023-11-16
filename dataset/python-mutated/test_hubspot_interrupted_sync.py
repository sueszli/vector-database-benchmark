from datetime import datetime, timedelta
from time import sleep
import copy
import tap_tester.connections as connections
import tap_tester.menagerie as menagerie
import tap_tester.runner as runner
from base import HubspotBaseTest
from client import TestClient

class TestHubspotInterruptedSync1(HubspotBaseTest):
    """Testing interrupted syncs for streams that implement unique bookmarking logic."""

    @staticmethod
    def name():
        if False:
            return 10
        return 'tt_hubspot_sync_interrupt_1'

    def streams_to_test(self):
        if False:
            return 10
        'expected streams minus the streams not under test'
        return {'companies', 'engagements'}

    def simulated_interruption(self, reference_state):
        if False:
            for i in range(10):
                print('nop')
        new_state = copy.deepcopy(reference_state)
        companies_bookmark = self.timedelta_formatted(reference_state['bookmarks']['companies']['property_hs_lastmodifieddate'], days=-1, str_format=self.BASIC_DATE_FORMAT)
        new_state['bookmarks']['companies']['property_hs_lastmodifieddate'] = None
        new_state['bookmarks']['companies']['current_sync_start'] = companies_bookmark
        engagements_bookmark = self.timedelta_formatted(reference_state['bookmarks']['engagements']['lastUpdated'], days=-1, str_format=self.BASIC_DATE_FORMAT)
        new_state['bookmarks']['engagements']['lastUpdated'] = None
        new_state['bookmarks']['engagements']['current_sync_start'] = engagements_bookmark
        return new_state

    def get_properties(self):
        if False:
            i = 10
            return i + 15
        return {'start_date': datetime.strftime(datetime.today() - timedelta(days=3), self.START_DATE_FORMAT)}

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.maxDiff = None

    def test_run(self):
        if False:
            while True:
                i = 10
        expected_streams = self.streams_to_test()
        conn_id = connections.ensure_connection(self)
        found_catalogs = self.run_and_verify_check_mode(conn_id)
        catalog_entries = [ce for ce in found_catalogs if ce['tap_stream_id'] in expected_streams]
        for catalog_entry in catalog_entries:
            stream_schema = menagerie.get_annotated_schema(conn_id, catalog_entry['stream_id'])
            connections.select_catalog_and_fields_via_metadata(conn_id, catalog_entry, stream_schema)
        first_record_count_by_stream = self.run_and_verify_sync(conn_id)
        synced_records = runner.get_records_from_target_output()
        state_1 = menagerie.get_state(conn_id)
        new_state = self.simulated_interruption(state_1)
        menagerie.set_state(conn_id, new_state)
        second_record_count_by_stream = self.run_and_verify_sync(conn_id)
        synced_records_2 = runner.get_records_from_target_output()
        state_2 = menagerie.get_state(conn_id)
        for stream in expected_streams:
            with self.subTest(stream=stream):
                replication_method = self.expected_replication_method()[stream]
                primary_keys = self.expected_primary_keys()[stream]
                actual_record_count_2 = second_record_count_by_stream[stream]
                actual_records_2 = [message['data'] for message in synced_records_2[stream]['messages'] if message['action'] == 'upsert']
                actual_record_count_1 = first_record_count_by_stream[stream]
                actual_records_1 = [message['data'] for message in synced_records[stream]['messages'] if message['action'] == 'upsert']
                if replication_method == self.INCREMENTAL:
                    stream_replication_key = list(self.expected_replication_keys()[stream])[0]
                    bookmark_1 = state_1['bookmarks'][stream][stream_replication_key]
                    bookmark_2 = state_2['bookmarks'][stream][stream_replication_key]
                    if stream in {'companies', 'engagements'}:
                        continue
                    self.assertEqual(bookmark_1, bookmark_2)
                    sync_1_pks = [tuple([record[pk] for pk in primary_keys]) for record in actual_records_1]
                    sync_2_pks = [tuple([record[pk] for pk in primary_keys]) for record in actual_records_2]
                    self.assertCountEqual(set(sync_1_pks), sync_1_pks)
                    self.assertCountEqual(set(sync_2_pks), sync_2_pks)
                    self.assertTrue(set(sync_2_pks).issubset(set(sync_1_pks)))
                else:
                    raise AssertionError(f'Replication method is {replication_method} for stream: {stream}')
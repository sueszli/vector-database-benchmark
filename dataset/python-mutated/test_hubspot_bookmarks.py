from datetime import datetime, timedelta
from time import sleep
import tap_tester.connections as connections
import tap_tester.menagerie as menagerie
import tap_tester.runner as runner
from base import HubspotBaseTest
from client import TestClient
STREAMS_WITHOUT_UPDATES = {'email_events', 'contacts_by_company', 'workflows'}
STREAMS_WITHOUT_CREATES = {'campaigns', 'owners'}

class TestHubspotBookmarks(HubspotBaseTest):
    """Ensure tap replicates new and upated records based on the replication method of a given stream.

    Create records for each stream. Run check mode, perform table and field selection, and run a sync.
    Create 1 record for each stream and update 1 record for each stream prior to running a  2nd sync.
     - Verify for each incremental stream you can do a sync which records bookmarks, and that the format matches expectations.
     - Verify that a bookmark doesn't exist for full table streams.
     - Verify the bookmark is the max value sent to the target for the a given replication key.
     - Verify 2nd sync respects the bookmark.
    """

    @staticmethod
    def name():
        if False:
            while True:
                i = 10
        return 'tt_hubspot_bookmarks'

    def streams_to_test(self):
        if False:
            i = 10
            return i + 15
        'expected streams minus the streams not under test'
        expected_streams = self.expected_streams().difference(STREAMS_WITHOUT_CREATES)
        return expected_streams.difference({'subscription_changes'})

    def get_properties(self):
        if False:
            print('Hello World!')
        return {'start_date': datetime.strftime(datetime.today() - timedelta(days=3), self.START_DATE_FORMAT)}

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.maxDiff = None
        self.test_client = TestClient(self.get_properties()['start_date'])

    def create_test_data(self, expected_streams):
        if False:
            while True:
                i = 10
        self.expected_records = {stream: [] for stream in expected_streams}
        for stream in expected_streams - {'contacts_by_company'}:
            if stream == 'email_events':
                email_records = self.test_client.create(stream, times=3)
                self.expected_records['email_events'] += email_records
            else:
                for _ in range(3):
                    record = self.test_client.create(stream)
                    self.expected_records[stream] += record
        if 'contacts_by_company' in expected_streams:
            company_ids = [record['companyId'] for record in self.expected_records['companies']]
            contact_records = self.expected_records['contacts']
            for i in range(3):
                record = self.test_client.create_contacts_by_company(company_ids=company_ids, contact_records=contact_records)
                self.expected_records['contacts_by_company'] += record

    def test_run(self):
        if False:
            print('Hello World!')
        expected_streams = self.streams_to_test()
        create_streams = expected_streams - STREAMS_WITHOUT_CREATES
        self.create_test_data(create_streams)
        conn_id = connections.ensure_connection(self)
        found_catalogs = self.run_and_verify_check_mode(conn_id)
        catalog_entries = [ce for ce in found_catalogs if ce['tap_stream_id'] in expected_streams]
        for catalog_entry in catalog_entries:
            stream_schema = menagerie.get_annotated_schema(conn_id, catalog_entry['stream_id'])
            connections.select_catalog_and_fields_via_metadata(conn_id, catalog_entry, stream_schema)
        first_record_count_by_stream = self.run_and_verify_sync(conn_id)
        synced_records = runner.get_records_from_target_output()
        state_1 = menagerie.get_state(conn_id)
        for stream in expected_streams - {'contacts_by_company'}:
            record = self.test_client.create(stream)
            self.expected_records[stream] += record
        if 'contacts_by_company' in expected_streams:
            company_ids = [record['companyId'] for record in self.expected_records['companies'][:-1]]
            contact_records = self.expected_records['contacts'][-1:]
            record = self.test_client.create_contacts_by_company(company_ids=company_ids, contact_records=contact_records)
            self.expected_records['contacts_by_company'] += record
        for stream in expected_streams - STREAMS_WITHOUT_UPDATES:
            primary_key = list(self.expected_primary_keys()[stream])[0]
            record_id = self.expected_records[stream][0][primary_key]
            record = self.test_client.update(stream, record_id)
            self.expected_records[stream].append(record)
        second_record_count_by_stream = self.run_and_verify_sync(conn_id)
        synced_records_2 = runner.get_records_from_target_output()
        state_2 = menagerie.get_state(conn_id)
        for stream in expected_streams:
            with self.subTest(stream=stream):
                replication_method = self.expected_replication_method()[stream]
                primary_keys = self.expected_primary_keys()[stream]
                expected_records_1 = self.expected_records[stream][:3]
                actual_record_count_2 = second_record_count_by_stream[stream]
                actual_records_2 = [message['data'] for message in synced_records_2[stream]['messages'] if message['action'] == 'upsert']
                actual_record_count_1 = first_record_count_by_stream[stream]
                actual_records_1 = [message['data'] for message in synced_records[stream]['messages'] if message['action'] == 'upsert']
                if self.is_child(stream):
                    parent_stream = self.expected_metadata()[stream][self.PARENT_STREAM]
                    parent_replication_method = self.expected_replication_method()[parent_stream]
                    if parent_replication_method == self.INCREMENTAL:
                        expected_record_count = 1 if stream not in STREAMS_WITHOUT_UPDATES else 2
                        expected_records_2 = self.expected_records[stream][-expected_record_count:]
                        self.assertGreater(actual_record_count_1, actual_record_count_2)
                    elif parent_replication_method == self.FULL:
                        expected_records_2 = self.expected_records[stream]
                        self.assertEqual(actual_record_count_1 + 1, actual_record_count_2)
                    else:
                        raise AssertionError(f'Replication method is {replication_method} for stream: {stream}')
                elif replication_method == self.INCREMENTAL:
                    stream_replication_key = list(self.expected_replication_keys()[stream])[0]
                    bookmark_1 = state_1['bookmarks'][stream][stream_replication_key]
                    bookmark_2 = state_2['bookmarks'][stream][stream_replication_key]
                    expected_record_count = 1 if stream not in STREAMS_WITHOUT_UPDATES else 2
                    expected_records_2 = self.expected_records[stream][-expected_record_count:]
                    if stream not in {'companies', 'deals', 'contacts_by_company', 'email_events'}:
                        for record in actual_records_1:
                            replication_key_value = record.get(stream_replication_key)
                            self.assertLessEqual(replication_key_value, bookmark_1, msg='First sync bookmark was incorrect, A record with greater replication-key value was found.')
                        for record in actual_records_2:
                            replication_key_value = record.get(stream_replication_key)
                            self.assertLessEqual(replication_key_value, bookmark_2, msg='Second sync bookmark was incorrect, A record with greater replication-key value was found.')
                    self.assertGreater(actual_record_count_1, actual_record_count_2)
                    if stream != 'email_events':
                        self.assertGreater(bookmark_2, bookmark_1)
                elif replication_method == self.FULL:
                    expected_records_2 = self.expected_records[stream]
                    self.assertEqual(actual_record_count_1 + 1, actual_record_count_2)
                else:
                    raise AssertionError(f'Replication method is {replication_method} for stream: {stream}')
                sync_1_pks = [tuple([record[pk] for pk in primary_keys]) for record in actual_records_1]
                expected_sync_1_pks = [tuple([record[pk] for pk in primary_keys]) for record in expected_records_1]
                for expected_pk in expected_sync_1_pks:
                    self.assertIn(expected_pk, sync_1_pks)
                sync_2_pks = sorted([tuple([record[pk] for pk in primary_keys]) for record in actual_records_2])
                expected_sync_2_pks = sorted([tuple([record[pk] for pk in primary_keys]) for record in expected_records_2])
                for expected_pk in expected_sync_2_pks:
                    self.assertIn(expected_pk, sync_2_pks)
                if stream in {'companies', 'email_events'}:
                    continue
                self.assertTrue(any([expected_pk in sync_2_pks for expected_pk in expected_sync_1_pks]))
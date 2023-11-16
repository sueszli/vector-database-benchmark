import tap_tester.connections as connections
import tap_tester.menagerie as menagerie
import tap_tester.runner as runner
import re
from base import HubspotBaseTest
STATIC_DATA_STREAMS = {'owners'}

class TestHubspotAutomaticFields(HubspotBaseTest):

    @staticmethod
    def name():
        if False:
            for i in range(10):
                print('nop')
        return 'tt_hubspot_automatic'

    def streams_to_test(self):
        if False:
            print('Hello World!')
        'streams to test'
        return self.expected_streams() - STATIC_DATA_STREAMS

    def test_run(self):
        if False:
            while True:
                i = 10
        '\n        Verify we can deselect all fields except when inclusion=automatic, which is handled by base.py methods\n        Verify that only the automatic fields are sent to the target.\n        '
        conn_id = connections.ensure_connection(self)
        found_catalogs = self.run_and_verify_check_mode(conn_id)
        expected_streams = self.streams_to_test()
        catalog_entries = [ce for ce in found_catalogs if ce['tap_stream_id'] in expected_streams]
        self.select_all_streams_and_fields(conn_id, catalog_entries, select_all_fields=False)
        sync_record_count = self.run_and_verify_sync(conn_id)
        synced_records = runner.get_records_from_target_output()
        for stream in expected_streams:
            with self.subTest(stream=stream):
                record_count = sync_record_count.get(stream, 0)
                self.assertGreater(record_count, 0)
                data = synced_records.get(stream)
                record_messages_keys = [set(row['data'].keys()) for row in data['messages']]
                expected_keys = self.expected_automatic_fields().get(stream)
                if stream in {'subscription_changes', 'email_events'}:
                    remove_keys = self.expected_metadata()[stream].get(self.REPLICATION_KEYS)
                    expected_keys = expected_keys.difference(remove_keys)
                elif stream in {'engagements'}:
                    expected_keys = expected_keys.union({'engagement'})
                for actual_keys in record_messages_keys:
                    self.assertSetEqual(actual_keys, expected_keys, msg=f'Expected automatic fields: {expected_keys} and nothing else.')
                if stream != 'subscription_changes':
                    pk = self.expected_primary_keys()[stream]
                    pks_values = [tuple([message['data'][p] for p in pk]) for message in data['messages']]
                    self.assertEqual(len(pks_values), len(set(pks_values)))

class TestHubspotAutomaticFieldsStaticData(TestHubspotAutomaticFields):

    def streams_to_test(self):
        if False:
            print('Hello World!')
        'streams to test'
        return STATIC_DATA_STREAMS

    @staticmethod
    def name():
        if False:
            for i in range(10):
                print('nop')
        return 'tt_hubspot_automatic_static'

    def get_properties(self):
        if False:
            while True:
                i = 10
        return {'start_date': '2021-08-19T00:00:00Z'}
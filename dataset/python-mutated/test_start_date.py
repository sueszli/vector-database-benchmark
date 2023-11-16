import tap_tester.connections as connections
import tap_tester.runner as runner
from base import ZendeskTest
from datetime import datetime

class ZendeskStartDate(ZendeskTest):
    """
    Ensure both all expected streams respect the start date. Run tap in check mode, 
    run 1st sync with start date = few days ago, run check mode and 2nd sync on a new connection with start date = today.
    """
    start_date_1 = ''
    start_date_2 = ''

    def name(self):
        if False:
            print('Hello World!')
        return 'zendesk_start_date_test'

    def test_run(self):
        if False:
            while True:
                i = 10
        '\n        Test that the start_date configuration is respected\n        • verify that a sync with a later start date has at least one record synced\n        and less records than the 1st sync with a previous start date\n        • verify that each stream has less records than the earlier start date sync\n        • verify all data from later start data has bookmark values >= start_date\n        '
        self.run_test(days=1172, expected_streams=self.expected_check_streams() - {'ticket_forms'})
        self.run_test(days=1774, expected_streams={'ticket_forms'})

    def run_test(self, days, expected_streams):
        if False:
            print('Hello World!')
        self.start_date_1 = self.get_properties().get('start_date')
        self.start_date_2 = self.timedelta_formatted(self.start_date_1, days=days)
        self.start_date = self.start_date_1
        expected_streams = expected_streams
        conn_id_1 = connections.ensure_connection(self)
        found_catalogs_1 = self.run_and_verify_check_mode(conn_id_1)
        test_catalogs_1_all_fields = [catalog for catalog in found_catalogs_1 if catalog.get('tap_stream_id') in expected_streams]
        self.perform_and_verify_table_and_field_selection(conn_id_1, test_catalogs_1_all_fields, select_all_fields=True)
        record_count_by_stream_1 = self.run_and_verify_sync(conn_id_1)
        synced_records_1 = runner.get_records_from_target_output()
        print('REPLICATION START DATE CHANGE: {} ===>>> {} '.format(self.start_date, self.start_date_2))
        self.start_date = self.start_date_2
        conn_id_2 = connections.ensure_connection(self, original_properties=False)
        found_catalogs_2 = self.run_and_verify_check_mode(conn_id_2)
        test_catalogs_2_all_fields = [catalog for catalog in found_catalogs_2 if catalog.get('tap_stream_id') in expected_streams]
        self.perform_and_verify_table_and_field_selection(conn_id_2, test_catalogs_2_all_fields, select_all_fields=True)
        record_count_by_stream_2 = self.run_and_verify_sync(conn_id_2)
        synced_records_2 = runner.get_records_from_target_output()
        for stream in expected_streams:
            with self.subTest(stream=stream):
                expected_primary_keys = self.expected_primary_keys()[stream]
                record_count_sync_1 = record_count_by_stream_1.get(stream, 0)
                record_count_sync_2 = record_count_by_stream_2.get(stream, 0)
                primary_keys_list_1 = [tuple((message.get('data').get(expected_pk) for expected_pk in expected_primary_keys)) for message in synced_records_1.get(stream, {}).get('messages', []) if message.get('action') == 'upsert']
                primary_keys_list_2 = [tuple((message.get('data').get(expected_pk) for expected_pk in expected_primary_keys)) for message in synced_records_2.get(stream, {}).get('messages', []) if message.get('action') == 'upsert']
                primary_keys_sync_1 = set(primary_keys_list_1)
                primary_keys_sync_2 = set(primary_keys_list_2)
                if self.expected_metadata()[stream][self.OBEYS_START_DATE]:
                    expected_replication_key = next(iter(self.expected_replication_keys().get(stream, [])))
                    replication_dates_1 = [row.get('data').get(expected_replication_key) for row in synced_records_1.get(stream, {'messages': []}).get('messages', []) if row.get('data')]
                    replication_dates_2 = [row.get('data').get(expected_replication_key) for row in synced_records_2.get(stream, {'messages': []}).get('messages', []) if row.get('data')]
                    for replication_date in replication_dates_1:
                        if stream == 'tickets':
                            replication_date = datetime.utcfromtimestamp(replication_date).strftime('%Y-%m-%dT%H:%M:%SZ')
                        self.assertGreaterEqual(self.parse_date(replication_date), self.parse_date(self.start_date_1), msg='Report pertains to a date prior to our start date.\n' + 'Sync start_date: {}\n'.format(self.start_date_1) + 'Record date: {} '.format(replication_date))
                    for replication_date in replication_dates_2:
                        if stream == 'tickets':
                            replication_date = datetime.utcfromtimestamp(replication_date).strftime('%Y-%m-%dT%H:%M:%SZ')
                        self.assertGreaterEqual(self.parse_date(replication_date), self.parse_date(self.start_date_2), msg='Report pertains to a date prior to our start date.\n' + 'Sync start_date: {}\n'.format(self.start_date_2) + 'Record date: {} '.format(replication_date))
                    self.assertGreater(record_count_sync_1, record_count_sync_2)
                    self.assertTrue(primary_keys_sync_2.issubset(primary_keys_sync_1))
                elif not stream in ['ticket_comments', 'ticket_audits', 'ticket_metrics']:
                    self.assertEqual(record_count_sync_2, record_count_sync_1)
                    self.assertSetEqual(primary_keys_sync_1, primary_keys_sync_2)
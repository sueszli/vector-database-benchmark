import os
from tap_tester import connections, runner, LOGGER
from base import FacebookBaseTest

class FacebookStartDateTest(FacebookBaseTest):
    start_date_1 = ''
    start_date_2 = ''

    @staticmethod
    def name():
        if False:
            return 10
        return 'tap_tester_facebook_start_date_test'

    def streams_to_test(self):
        if False:
            return 10
        return self.expected_streams()

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        'Instantiate start date according to the desired data set and run the test'
        self.start_date_1 = '2021-04-07T00:00:00Z'
        self.start_date_2 = self.timedelta_formatted(self.start_date_1, days=2, date_format=self.START_DATE_FORMAT)
        self.start_date = self.start_date_1
        expected_streams = self.streams_to_test()
        conn_id_1 = connections.ensure_connection(self)
        found_catalogs_1 = self.run_and_verify_check_mode(conn_id_1)
        test_catalogs_1_all_fields = [catalog for catalog in found_catalogs_1 if catalog.get('tap_stream_id') in expected_streams]
        self.perform_and_verify_table_and_field_selection(conn_id_1, test_catalogs_1_all_fields, select_all_fields=True)
        record_count_by_stream_1 = self.run_and_verify_sync(conn_id_1)
        synced_records_1 = runner.get_records_from_target_output()
        LOGGER.info('REPLICATION START DATE CHANGE: %s ===>>> %s ', self.start_date, self.start_date_2)
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
                expected_insights_buffer = -1 * int(self.get_properties()['insights_buffer_days'])
                expected_start_date_1 = self.timedelta_formatted(self.start_date_1, days=expected_insights_buffer, date_format=self.START_DATE_FORMAT)
                expected_start_date_2 = self.timedelta_formatted(self.start_date_2, days=expected_insights_buffer, date_format=self.START_DATE_FORMAT)
                record_count_sync_1 = record_count_by_stream_1.get(stream, 0)
                record_count_sync_2 = record_count_by_stream_2.get(stream, 0)
                primary_keys_list_1 = [tuple((message.get('data').get(expected_pk) for expected_pk in expected_primary_keys)) for message in synced_records_1.get(stream).get('messages') if message.get('action') == 'upsert']
                primary_keys_list_2 = [tuple((message.get('data').get(expected_pk) for expected_pk in expected_primary_keys)) for message in synced_records_2.get(stream).get('messages') if message.get('action') == 'upsert']
                primary_keys_sync_1 = set(primary_keys_list_1)
                primary_keys_sync_2 = set(primary_keys_list_2)
                if self.is_insight(stream):
                    expected_replication_key = next(iter(self.expected_replication_keys().get(stream)))
                    replication_dates_1 = [row.get('data').get(expected_replication_key) for row in synced_records_1.get(stream, {'messages': []}).get('messages', []) if row.get('data')]
                    replication_dates_2 = [row.get('data').get(expected_replication_key) for row in synced_records_2.get(stream, {'messages': []}).get('messages', []) if row.get('data')]
                    for replication_date in replication_dates_1:
                        self.assertGreaterEqual(self.parse_date(replication_date), self.parse_date(expected_start_date_1), msg='Report pertains to a date prior to our start date.\n' + 'Sync start_date: {}\n'.format(expected_start_date_1) + 'Record date: {} '.format(replication_date))
                    for replication_date in replication_dates_2:
                        self.assertGreaterEqual(self.parse_date(replication_date), self.parse_date(expected_start_date_2), msg='Report pertains to a date prior to our start date.\n' + 'Sync start_date: {}\n'.format(expected_start_date_2) + 'Record date: {} '.format(replication_date))
                    self.assertGreater(record_count_sync_1, record_count_sync_2)
                    self.assertTrue(primary_keys_sync_2.issubset(primary_keys_sync_1))
                else:
                    self.assertEqual(record_count_sync_2, record_count_sync_1)
                    self.assertSetEqual(primary_keys_sync_1, primary_keys_sync_2)
from tap_tester import menagerie, connections, runner
import re
from base import FreshdeskBaseTest

class PaginationTest(FreshdeskBaseTest):

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'tap_freshdesk_pagination_test'

    def test_name(self):
        if False:
            for i in range(10):
                print('nop')
        print('Pagination Test for tap-freshdesk')

    def test_run(self):
        if False:
            while True:
                i = 10
        conn_id = connections.ensure_connection(self)
        streams_to_test = {'agents', 'tickets'}
        check_job_name = self.run_and_verify_check_mode(conn_id)
        sync_record_count = self.run_and_verify_sync(conn_id)
        sync_records = runner.get_records_from_target_output()
        for stream in streams_to_test:
            with self.subTest(stream=stream):
                record_count = sync_record_count.get(stream, 0)
                sync_messages = sync_records.get(stream, {'messages': []}).get('messages')
                primary_keys = self.expected_primary_keys().get(stream)
                stream_page_size = self.expected_page_limits()[stream]
                self.assertLess(stream_page_size, record_count)
                print('stream_page_size: {} < record_count {} for stream: {}'.format(stream_page_size, record_count, stream))
                records_pks_set = {tuple([message.get('data').get(primary_key) for primary_key in primary_keys]) for message in sync_messages}
                records_pks_list = [tuple([message.get('data').get(primary_key) for primary_key in primary_keys]) for message in sync_messages]
                self.assertCountEqual(records_pks_set, records_pks_list, msg=f'We have duplicate records for {stream}')
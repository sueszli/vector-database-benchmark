from datetime import datetime
from datetime import timedelta
import time
import tap_tester.connections as connections
import tap_tester.menagerie as menagerie
import tap_tester.runner as runner
from tap_tester.logger import LOGGER
from client import TestClient
from base import HubspotBaseTest

class TestHubspotPagination(HubspotBaseTest):

    @staticmethod
    def name():
        if False:
            for i in range(10):
                print('nop')
        return 'tt_hubspot_pagination'

    def get_properties(self):
        if False:
            while True:
                i = 10
        return {'start_date': datetime.strftime(datetime.today() - timedelta(days=7), self.START_DATE_FORMAT)}

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.maxDiff = None
        setup_start = time.perf_counter()
        test_client = TestClient(self.get_properties()['start_date'])
        existing_records = dict()
        limits = self.expected_page_limits()
        streams = self.streams_to_test()
        if 'subscription_changes' in streams and 'email_events' in streams:
            streams.remove('email_events')
        stream_to_run_last = 'contacts_by_company'
        if stream_to_run_last in streams:
            streams.remove(stream_to_run_last)
            streams = list(streams)
            streams.append(stream_to_run_last)
        for stream in streams:
            if stream == 'contacts_by_company':
                company_ids = [company['companyId'] for company in existing_records['companies']]
                existing_records[stream] = test_client.read(stream, parent_ids=company_ids)
            elif stream in {'companies', 'contact_lists', 'subscription_changes', 'engagements', 'email_events'}:
                existing_records[stream] = test_client.read(stream)
            else:
                existing_records[stream] = test_client.read(stream)
            LOGGER.info(f'Pagination limit set to - {limits[stream]} and total number of existing record - {len(existing_records[stream])}')
            under_target = limits[stream] + 1 - len(existing_records[stream])
            LOGGER.info(f'under_target = {under_target} for {stream}')
            if under_target > 0:
                LOGGER.info(f'need to make {under_target} records for {stream} stream')
                if stream in {'subscription_changes', 'emails_events'}:
                    test_client.create(stream, subscriptions=existing_records[stream], times=under_target)
                elif stream == 'contacts_by_company':
                    test_client.create(stream, company_ids, times=under_target)
                else:
                    for i in range(under_target):
                        test_client.create(stream)
        setup_end = time.perf_counter()
        LOGGER.info(f"Test Client took about {str(setup_end - setup_start).split('.')[0]} seconds")

    def streams_to_test(self):
        if False:
            while True:
                i = 10
        '\n        All streams with limits are under test\n        '
        streams_with_page_limits = {stream for (stream, limit) in self.expected_page_limits().items() if limit}
        streams_to_test = streams_with_page_limits.difference({'contacts_by_company', 'email_events', 'subscription_changes'})
        return streams_to_test

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        expected_streams = self.streams_to_test()
        conn_id = connections.ensure_connection(self)
        found_catalogs = self.run_and_verify_check_mode(conn_id)
        catalog_entries = [ce for ce in found_catalogs if ce['tap_stream_id'] in expected_streams]
        for catalog_entry in catalog_entries:
            stream_schema = menagerie.get_annotated_schema(conn_id, catalog_entry['stream_id'])
            connections.select_catalog_and_fields_via_metadata(conn_id, catalog_entry, stream_schema)
        sync_record_count = self.run_and_verify_sync(conn_id)
        sync_records = runner.get_records_from_target_output()
        for stream in expected_streams:
            with self.subTest(stream=stream):
                record_count = sync_record_count.get(stream, 0)
                sync_messages = sync_records.get(stream, {'messages': []}).get('messages')
                primary_keys = self.expected_primary_keys().get(stream)
                stream_page_size = self.expected_page_limits()[stream]
                self.assertLess(stream_page_size, record_count)
                records_pks_set = {tuple([message.get('data').get(primary_key) for primary_key in primary_keys]) for message in sync_messages}
                records_pks_list = [tuple([message.get('data').get(primary_key) for primary_key in primary_keys]) for message in sync_messages]
                self.assertCountEqual(records_pks_set, records_pks_list, msg=f'We have duplicate records for {stream}')
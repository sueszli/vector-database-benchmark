import re
import os
import pytz
import time
import dateutil.parser
from datetime import timedelta
from datetime import datetime
from tap_tester import menagerie, connections, runner
from base import FreshdeskBaseTest

class FreshdeskBookmarks(FreshdeskBaseTest):
    """Test incremental replication via bookmarks (without CRUD)."""
    start_date = ''
    test_streams = {}

    @staticmethod
    def name():
        if False:
            while True:
                i = 10
        return 'tt_freshdesk_bookmarks'

    def get_properties(self):
        if False:
            i = 10
            return i + 15
        return_value = {'start_date': '2019-01-04T00:00:00Z'}
        self.start_date = return_value['start_date']
        return return_value

    def calculated_states_by_stream(self, current_state):
        if False:
            while True:
                i = 10
        '\n        Look at the bookmarks from a previous sync and set a new bookmark\n        value based off timedelta expectations. This ensures the subsequent sync will replicate\n        at least 1 record but, fewer records than the previous sync.\n\n        Sufficient test data is required for this test to cover a given stream.\n        An incremental replication stream must have at least two records with\n        replication keys that differ by some time span.\n\n        If the test data is changed in the future this may break expectations for this test.\n        '
        bookmark_streams = self.test_streams - {'conversations'}
        print('bookmark_streams: {}'.format(bookmark_streams))
        timedelta_by_stream = {stream: [0, 12, 0] for stream in bookmark_streams}
        current_state = {'bookmarks': current_state}
        del current_state['bookmarks']['tickets_deleted']
        del current_state['bookmarks']['tickets_spam']
        stream_to_calculated_state = {stream: '' for stream in bookmark_streams}
        for (stream, state_value) in current_state['bookmarks'].items():
            if stream in bookmark_streams:
                state_as_datetime = dateutil.parser.parse(state_value)
                (days, hours, minutes) = timedelta_by_stream[stream]
                calculated_state_as_datetime = state_as_datetime - timedelta(days=days, hours=hours, minutes=minutes)
                state_format = self.BOOKMARK_FORMAT
                calculated_state_formatted = datetime.strftime(calculated_state_as_datetime, state_format)
                if calculated_state_formatted < self.start_date:
                    raise RuntimeError('Time delta error for stream {}, sim start_date < start_date!'.format(stream))
                stream_to_calculated_state[stream] = calculated_state_formatted
        return stream_to_calculated_state

    def test_run(self):
        if False:
            print('Hello World!')
        'A Bookmarks Test'
        self.test_streams = {'tickets', 'companies', 'agents', 'groups', 'roles', 'conversations'}
        expected_replication_keys = self.expected_replication_keys()
        expected_replication_methods = self.expected_replication_method()
        conn_id = connections.ensure_connection(self)
        check_job_name = self.run_and_verify_check_mode(conn_id)
        first_sync_record_count = self.run_and_verify_sync(conn_id)
        first_sync_messages = runner.get_records_from_target_output()
        first_sync_bookmarks = menagerie.get_state(conn_id)
        first_sync_empty = self.test_streams - first_sync_messages.keys()
        if len(first_sync_empty) > 0:
            print('Missing stream(s): {} in sync 1. Failing test for stream(s)'.format(first_sync_empty))
        self.first_sync_empty = first_sync_empty
        first_sync_bonus = first_sync_messages.keys() - self.test_streams
        if len(first_sync_bonus) > 0:
            print('Found stream: {} in first sync. Add to test_streams?'.format(first_sync_bonus))
        simulated_states = self.calculated_states_by_stream(first_sync_bookmarks)
        menagerie.set_state(conn_id, simulated_states)
        second_sync_record_count = self.run_and_verify_sync(conn_id)
        second_sync_messages = runner.get_records_from_target_output()
        second_sync_bookmarks = menagerie.get_state(conn_id)
        second_sync_empty = self.test_streams - second_sync_messages.keys()
        if len(second_sync_empty) > 0:
            print('Missing stream(s): {} in sync 2. Failing test. Check test data!'.format(second_sync_empty))
        self.second_sync_empty = second_sync_empty
        second_sync_bonus = second_sync_messages.keys() - self.test_streams
        if len(second_sync_bonus) > 0:
            print('Found stream(s): {} in second sync. Add to test_streams?'.format(second_sync_bonus))
        for stream in self.test_streams:
            with self.subTest(stream=stream):
                if stream in self.first_sync_empty:
                    self.assertTrue(False, msg='Stream: {} no longer in sync 1. Check test data'.format(stream))
                    continue
                if stream in self.second_sync_empty:
                    if stream == 'conversations':
                        print('Commented out failing test case for stream: {}'.format(stream))
                        print('See https://jira.talendforge.org/browse/TDL-17738 for details')
                        continue
                    self.assertTrue(False, msg='Stream: {} present in sync 1, missing in sync 2!'.format(stream))
                    continue
                expected_replication_method = expected_replication_methods[stream]
                first_sync_count = first_sync_record_count.get(stream, 0)
                second_sync_count = second_sync_record_count.get(stream, 0)
                first_sync_records = [record.get('data') for record in first_sync_messages.get(stream).get('messages') if record.get('action') == 'upsert']
                second_sync_records = [record.get('data') for record in second_sync_messages.get(stream).get('messages') if record.get('action') == 'upsert']
                if stream != 'conversations':
                    first_bookmark_value = first_sync_bookmarks.get(stream)
                    second_bookmark_value = second_sync_bookmarks.get(stream)
                if expected_replication_method == self.INCREMENTAL:
                    replication_key = next(iter(expected_replication_keys[stream]))
                    if stream != 'conversations':
                        simulated_bookmark_value = simulated_states[stream]
                    if stream == 'conversations':
                        print('*** Only checking sync counts for stream: {}'.format(stream))
                        self.assertLessEqual(second_sync_count, first_sync_count)
                        if second_sync_count == first_sync_count:
                            print('WARN: first_sync_count == second_sync_count for stream: {}'.format(stream))
                        continue
                    self.assertIsNotNone(first_bookmark_value)
                    self.assertIsNotNone(second_bookmark_value)
                    self.assertEqual(second_bookmark_value, first_bookmark_value)
                    if stream == 'roles':
                        self.assertEqual(second_sync_count, first_sync_count)
                        print('WARN: Less covereage, unable to update records for stream: {}'.format(stream))
                    else:
                        self.assertLess(second_sync_count, first_sync_count)
                    rec_time = []
                    for record in first_sync_records:
                        rec_time += (record['updated_at'],)
                    rec_time.sort()
                    self.assertEqual(rec_time[-1], first_bookmark_value)
                    rec_time = []
                    for record in second_sync_records:
                        rec_time += (record['updated_at'],)
                    rec_time.sort()
                    self.assertEqual(rec_time[-1], second_bookmark_value)
                    for record in second_sync_records:
                        self.assertTrue(record['updated_at'] >= simulated_states[stream], msg='record time cannot be less than bookmark time')
                else:
                    raise NotImplementedError('INVALID EXPECTATIONS\t\tSTREAM: {} REPLICATION_METHOD: {}'.format(stream, expected_replication_method))
                self.assertGreater(second_sync_count, 0, msg='We are not fully testing bookmarking for {}'.format(stream))
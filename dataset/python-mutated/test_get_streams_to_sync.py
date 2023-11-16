import unittest
from tap_hubspot import get_streams_to_sync, parse_source_from_url, Stream

class TestGetStreamsToSync(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.streams = [Stream('a', 'a', [], None, None), Stream('b', 'b', [], None, None), Stream('c', 'c', [], None, None)]

    def test_get_streams_to_sync_with_no_this_stream(self):
        if False:
            for i in range(10):
                print('nop')
        state = {'this_stream': None}
        self.assertEqual(self.streams, get_streams_to_sync(self.streams, state))

    def test_get_streams_to_sync_with_first_stream(self):
        if False:
            while True:
                i = 10
        state = {'currently_syncing': 'a'}
        result = get_streams_to_sync(self.streams, state)
        parsed_result = [s.tap_stream_id for s in result]
        self.assertEqual(parsed_result, ['a', 'b', 'c'])

    def test_get_streams_to_sync_with_middle_stream(self):
        if False:
            print('Hello World!')
        state = {'currently_syncing': 'b'}
        result = get_streams_to_sync(self.streams, state)
        parsed_result = [s.tap_stream_id for s in result]
        self.assertEqual(parsed_result, ['b', 'c', 'a'])

    def test_get_streams_to_sync_with_last_stream(self):
        if False:
            print('Hello World!')
        state = {'currently_syncing': 'c'}
        result = get_streams_to_sync(self.streams, state)
        parsed_result = [s.tap_stream_id for s in result]
        self.assertEqual(parsed_result, ['c', 'a', 'b'])

    def test_parse_source_from_url_succeeds(self):
        if False:
            while True:
                i = 10
        url = 'https://api.hubapi.com/companies/v2/companies/recent/modified'
        self.assertEqual('companies', parse_source_from_url(url))
import os
from tap_tester import runner, connections, menagerie
from base import TestLinkedinAdsBase

class AllFields(TestLinkedinAdsBase):
    """Test that with all fields selected for a stream automatic and available fields are  replicated"""
    access_token = None

    @staticmethod
    def set_access_token(access_token):
        if False:
            return 10
        AllFields.access_token = access_token

    def get_credentials(self):
        if False:
            for i in range(10):
                print('nop')
        return {'access_token': AllFields.access_token}

    def run_all_fields(self):
        if False:
            while True:
                i = 10
        '\n        Ensure running the tap with all streams and fields selected results in the\n        replication of all fields.\n        - Verify no unexpected streams were replicated\n        - Verify that more than just the automatic fields are replicated for each stream.\n        '
        expected_streams = self.expected_streams()
        conn_id = connections.ensure_connection(self)
        found_catalogs = self.run_and_verify_check_mode(conn_id)
        test_catalogs_all_fields = [catalog for catalog in found_catalogs if catalog.get('stream_name') in expected_streams]
        self.perform_and_verify_table_and_field_selection(conn_id, test_catalogs_all_fields, select_all_fields=True)
        stream_to_all_catalog_fields = dict()
        for catalog in test_catalogs_all_fields:
            (stream_id, stream_name) = (catalog['stream_id'], catalog['stream_name'])
            catalog_entry = menagerie.get_annotated_schema(conn_id, stream_id)
            fields_from_field_level_md = [md_entry['breadcrumb'][1] for md_entry in catalog_entry['metadata'] if md_entry['breadcrumb'] != []]
            stream_to_all_catalog_fields[stream_name] = set(fields_from_field_level_md)
        record_count_by_stream = self.run_and_verify_sync(conn_id)
        synced_records = runner.get_records_from_target_output()
        synced_stream_names = set(synced_records.keys())
        self.assertSetEqual(expected_streams, synced_stream_names)
        for stream in expected_streams:
            with self.subTest(stream=stream):
                expected_automatic_keys = self.expected_automatic_fields().get(stream)
                expected_all_keys = stream_to_all_catalog_fields[stream]
                messages = synced_records.get(stream)
                actual_all_keys = [set(message['data'].keys()) for message in messages['messages'] if message['action'] == 'upsert'][0]
                self.assertGreater(record_count_by_stream.get(stream, -1), 0)
                self.assertGreater(len(expected_all_keys), len(expected_automatic_keys))
                self.assertTrue(expected_automatic_keys.issubset(expected_all_keys), msg=f'{expected_automatic_keys - expected_all_keys} is not in "expected_all_keys"')
                if stream == 'creatives':
                    expected_all_keys.remove('reference_share_id')
                elif stream == 'campaigns':
                    expected_all_keys.remove('associated_entity_person_id')
                    expected_all_keys.remove('targeting')
                elif stream == 'video_ads':
                    expected_all_keys.remove('content_reference_share_id')
                    expected_all_keys.remove('content_reference_ucg_post_id')
                elif stream == 'accounts':
                    expected_all_keys.remove('total_budget_ends_at')
                    expected_all_keys.remove('total_budget')
                    expected_all_keys.remove('reference_person_id')
                elif stream in ['ad_analytics_by_creative', 'ad_analytics_by_campaign']:
                    expected_all_keys.remove('lead_generation_mail_interest_clicks')
                self.assertSetEqual(expected_all_keys, actual_all_keys)

class AllFieldsWithExpiredAccessToken(AllFields):
    """This method run all fileds test by setting expired access token in the config properties"""

    @staticmethod
    def name():
        if False:
            print('Hello World!')
        return 'tap_tester_linkedin_expired_access_token'

    def test_run(self):
        if False:
            return 10
        try:
            self.set_access_token(os.getenv('TAP_LINKEDIN_ADS_EXPIRED_ACCESS_TOKEN', None))
            self.run_all_fields()
        except Exception as e:
            self.assertIn('HTTP-error-code: 401, Error: The token used in the request has expired', str(e))

class AllFieldsWithInvalidAccessToken(AllFields):
    """This method run all fileds test by setting invalid access token in the config properties"""

    @staticmethod
    def name():
        if False:
            return 10
        return 'tap_tester_linkedin_invalid_access_token'

    def test_run(self):
        if False:
            return 10
        try:
            self.set_access_token('INVALID_ACCESS_TOKEN')
            self.run_all_fields()
        except Exception as e:
            self.assertIn('HTTP-error-code: 401, Error: Invalid access token', str(e))
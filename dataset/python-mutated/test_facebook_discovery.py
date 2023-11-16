"""Test tap discovery mode and metadata."""
import re
from tap_tester import menagerie, connections
from base import FacebookBaseTest

class DiscoveryTest(FacebookBaseTest):
    """Test tap discovery mode and metadata conforms to standards."""

    @staticmethod
    def name():
        if False:
            i = 10
            return i + 15
        return 'tap_tester_facebook_discovery_test'

    def streams_to_test(self):
        if False:
            i = 10
            return i + 15
        return self.expected_streams()

    def test_run(self):
        if False:
            return 10
        '\n        Testing that discovery creates the appropriate catalog with valid metadata.\n\n        • Verify number of actual streams discovered match expected\n        • Verify the stream names discovered were what we expect\n        • Verify stream names follow naming convention\n          streams should only have lowercase alphas and underscores\n        • verify there is only 1 top level breadcrumb\n        • verify replication key(s)\n        • verify primary key(s)\n        • verify that if there is a replication key we are doing INCREMENTAL otherwise FULL\n        • verify the actual replication matches our expected replication method\n        • verify that primary, replication and foreign keys\n          are given the inclusion of automatic.\n        • verify that all other fields have inclusion of available metadata.\n        '
        streams_to_test = self.streams_to_test()
        conn_id = connections.ensure_connection(self)
        found_catalogs = self.run_and_verify_check_mode(conn_id)
        found_catalog_names = {c['tap_stream_id'] for c in found_catalogs}
        self.assertTrue(all([re.fullmatch('[a-z_]+', name) for name in found_catalog_names]), msg="One or more streams don't follow standard naming")
        for stream in streams_to_test:
            with self.subTest(stream=stream):
                catalog = next(iter([catalog for catalog in found_catalogs if catalog['stream_name'] == stream]))
                self.assertIsNotNone(catalog)
                expected_primary_keys = self.expected_primary_keys()[stream]
                expected_replication_keys = self.expected_replication_keys()[stream]
                expected_automatic_fields = expected_primary_keys | expected_replication_keys
                expected_replication_method = self.expected_replication_method()[stream]
                schema_and_metadata = menagerie.get_annotated_schema(conn_id, catalog['stream_id'])
                metadata = schema_and_metadata['metadata']
                stream_properties = [item for item in metadata if item.get('breadcrumb') == []]
                actual_primary_keys = set(stream_properties[0].get('metadata', {self.PRIMARY_KEYS: []}).get(self.PRIMARY_KEYS, []))
                actual_replication_keys = set(stream_properties[0].get('metadata', {self.REPLICATION_KEYS: []}).get(self.REPLICATION_KEYS, []))
                actual_replication_method = stream_properties[0].get('metadata', {self.REPLICATION_METHOD: None}).get(self.REPLICATION_METHOD)
                actual_automatic_fields = set((item.get('breadcrumb', ['properties', None])[1] for item in metadata if item.get('metadata').get('inclusion') == 'automatic'))
                self.assertTrue(len(stream_properties) == 1, msg='There is NOT only one top level breadcrumb for {}'.format(stream) + '\nstream_properties | {}'.format(stream_properties))
                self.assertSetEqual(expected_replication_keys, actual_replication_keys)
                self.assertSetEqual(expected_primary_keys, actual_primary_keys)
                self.assertEqual(expected_replication_method, actual_replication_method)
                if actual_replication_keys:
                    self.assertEqual(self.INCREMENTAL, actual_replication_method)
                else:
                    self.assertEqual(self.FULL_TABLE, actual_replication_method)
                self.assertSetEqual(expected_automatic_fields, actual_automatic_fields)
                self.assertTrue(all({item.get('metadata').get('inclusion') == 'available' for item in metadata if item.get('breadcrumb', []) != [] and item.get('breadcrumb', ['properties', None])[1] not in actual_automatic_fields}), msg='Not all non key properties are set to available in metadata')
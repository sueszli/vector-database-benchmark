import tap_tester.connections as connections
import tap_tester.runner as runner
import tap_tester.menagerie as menagerie
from base import ZendeskTest

class ZendeskAllFields(ZendeskTest):
    """Ensure running the tap with all streams and fields selected results in the replication of all fields."""

    def name(self):
        if False:
            return 10
        return 'zendesk_all_fields'

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        • Verify no unexpected streams were replicated\n        • Verify that more than just the automatic fields are replicated for each stream. \n        • verify all fields for each stream are replicated\n        '
        expected_streams = self.expected_check_streams()
        expected_automatic_fields = self.expected_automatic_fields()
        conn_id = connections.ensure_connection(self)
        found_catalogs = self.run_and_verify_check_mode(conn_id)
        test_catalogs_all_fields = [catalog for catalog in found_catalogs if catalog.get('tap_stream_id') in expected_streams]
        self.perform_and_verify_table_and_field_selection(conn_id, test_catalogs_all_fields)
        stream_to_all_catalog_fields = dict()
        for catalog in test_catalogs_all_fields:
            (stream_id, stream_name) = (catalog['stream_id'], catalog['stream_name'])
            catalog_entry = menagerie.get_annotated_schema(conn_id, stream_id)
            fields_from_field_level_md = [md_entry['breadcrumb'][1] for md_entry in catalog_entry['metadata'] if md_entry['breadcrumb'] != []]
            stream_to_all_catalog_fields[stream_name] = set(fields_from_field_level_md)
        self.run_and_verify_sync(conn_id)
        synced_records = runner.get_records_from_target_output()
        synced_stream_names = set(synced_records.keys())
        self.assertSetEqual(expected_streams, synced_stream_names)
        for stream in expected_streams:
            with self.subTest(stream=stream):
                expected_all_keys = stream_to_all_catalog_fields[stream]
                expected_automatic_keys = expected_automatic_fields.get(stream, set())
                self.assertTrue(expected_automatic_keys.issubset(expected_all_keys), msg='{} is not in "expected_all_keys"'.format(expected_automatic_keys - expected_all_keys))
                messages = synced_records.get(stream)
                actual_all_keys = set()
                for message in messages['messages']:
                    if message['action'] == 'upsert':
                        actual_all_keys.update(message['data'].keys())
                if stream == 'ticket_fields':
                    expected_all_keys = expected_all_keys - {'system_field_options', 'sub_type_id'}
                elif stream == 'users':
                    expected_all_keys = expected_all_keys - {'permanently_deleted'}
                elif stream == 'ticket_metrics':
                    expected_all_keys = expected_all_keys - {'status', 'instance_id', 'metric', 'type', 'time'}
                self.assertSetEqual(expected_all_keys, actual_all_keys)
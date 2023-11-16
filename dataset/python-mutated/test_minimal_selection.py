import tap_tester.connections as connections
import tap_tester.menagerie as menagerie
import tap_tester.runner as runner
from base import ZendeskTest
import unittest
from functools import reduce
from singer import metadata

class ZendeskMinimalSelection(ZendeskTest):

    def name(self):
        if False:
            print('Hello World!')
        return 'tap_tester_zendesk_minimal_selection'

    def expected_sync_streams(self):
        if False:
            for i in range(10):
                print('nop')
        return {'users'}

    def expected_pks(self):
        if False:
            return 10
        return {'users': {'id'}}

    def test_run(self):
        if False:
            i = 10
            return i + 15
        conn_id = connections.ensure_connection(self)
        check_job_name = runner.run_check_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, check_job_name)
        menagerie.verify_check_exit_status(self, exit_status, check_job_name)
        self.found_catalogs = menagerie.get_catalogs(conn_id)
        self.assertEqual(len(self.found_catalogs), len(self.expected_check_streams()))
        found_catalog_names = {catalog['tap_stream_id'] for catalog in self.found_catalogs if catalog['tap_stream_id'] in self.expected_check_streams()}
        self.assertSetEqual(self.expected_check_streams(), found_catalog_names)
        our_catalogs = [c for c in self.found_catalogs if c.get('tap_stream_id') in self.expected_sync_streams()]
        for c in our_catalogs:
            c_annotated = menagerie.get_annotated_schema(conn_id, c['stream_id'])
            c_metadata = metadata.to_map(c_annotated['metadata'])
            connections.select_catalog_and_fields_via_metadata(conn_id, c, c_annotated, [], ['name'])
        menagerie.set_state(conn_id, {})
        _ = self.run_and_verify_sync(conn_id)
        records = runner.get_records_from_target_output()
        for stream in self.expected_sync_streams():
            messages = records.get(stream).get('messages')
            for m in messages:
                pk_set = self.expected_pks()[stream]
                for pk in pk_set:
                    self.assertIsNotNone(m.get('data', {}).get(pk), msg='Missing primary-key for message {}'.format(m))
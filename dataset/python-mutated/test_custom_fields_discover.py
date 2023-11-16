from tap_tester import menagerie, connections, runner
from base import ZendeskTest

class ZendeskCustomFieldsDiscover(ZendeskTest):

    def name(self):
        if False:
            while True:
                i = 10
        return 'tap_tester_zendesk_custom_fields_discover'

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        conn_id = connections.ensure_connection(self)
        check_job_name = runner.run_check_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, check_job_name)
        menagerie.verify_check_exit_status(self, exit_status, check_job_name)
        self.found_catalogs = menagerie.get_catalogs(conn_id)
        self.assertEqual(len(self.found_catalogs), len(self.expected_check_streams()))
        found_catalog_names = {catalog['tap_stream_id'] for catalog in self.found_catalogs if catalog['tap_stream_id'] in self.expected_check_streams()}
        self.assertSetEqual(self.expected_check_streams(), found_catalog_names)
        streams = [c for c in self.found_catalogs if c['stream_name'] in ['organizations', 'users']]
        schemas = [(s['stream_name'][:-1], menagerie.get_annotated_schema(conn_id, s['stream_id'])) for s in streams]
        for schema in schemas:
            properties = schema[1]['annotated-schema']['properties']
            self.assertIsNotNone(properties.get('{}_fields'.format(schema[0]), {}).get('properties'), msg='{}_fields not present in schema!'.format(schema[0]))
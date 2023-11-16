import os
import tap_tester.connections as connections
import tap_tester.menagerie as menagerie
import tap_tester.runner as runner
from functools import reduce
from zenpy import Zenpy
from zenpy.lib.api_objects import Group, Organization, Tag, User
from base import ZendeskTest

class ZendeskAllStreams(ZendeskTest):

    def name(self):
        if False:
            return 10
        return 'tap_tester_zendesk_all_streams'

    def expected_sync_streams(self):
        if False:
            i = 10
            return i + 15
        return {'tickets', 'groups', 'users', 'organizations', 'ticket_audits', 'ticket_fields', 'group_memberships', 'macros', 'tags', 'ticket_metrics'}

    def expected_pks(self):
        if False:
            return 10
        return {'tickets': {'id'}, 'groups': {'id'}, 'users': {'id'}, 'organizations': {'id'}, 'ticket_audits': {'id'}, 'ticket_fields': {'id'}, 'group_memberships': {'id'}, 'macros': {'id'}, 'tags': {'name'}, 'ticket_metrics': {'id'}}

    def refresh_tags(self, records):
        if False:
            i = 10
            return i + 15
        creds = {'email': 'dev@stitchdata.com', 'password': os.getenv('TAP_ZENDESK_API_PASSWORD'), 'subdomain': 'rjmdev'}
        test_tags = ['test_tag_1', 'test_tag_2', 'test_tag_3']
        unclosed_tickets = [t for t in records.get('tickets').get('messages') if t.get('data').get('status') != 'closed']
        self.assertGreaterEqual(len(unclosed_tickets), 3)
        last_3_unclosed_tickets = unclosed_tickets[-3:]
        zenpy_client = Zenpy(**creds)
        for (i, tic) in enumerate(last_3_unclosed_tickets):
            if tic.get('data').get('tags'):
                tag_list = tic.get('data').get('tags')
                zenpy_client.tickets.delete_tags(tic.get('data').get('id'), tag_list)
                zenpy_client.tickets.add_tags(tic.get('data').get('id'), tag_list)
            zenpy_client.tickets.add_tags(tic.get('data').get('id'), test_tags[0:i + 1])
            self.tags_are_stale = False

    def test_run(self):
        if False:
            while True:
                i = 10
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
            c_metadata = self.to_map(c_annotated['metadata'])
            connections.select_catalog_and_fields_via_metadata(conn_id, c, c_annotated, [], [])
        menagerie.set_state(conn_id, {})
        _ = self.run_and_verify_sync(conn_id, state={})
        records = runner.get_records_from_target_output()
        self.tags_are_stale = True
        if not records.get('tags').get('messages', []):
            self.refresh_tags(records)
            _ = self.run_and_verify_sync(conn_id)
            tags_records = runner.get_records_from_target_output()
            self.assertGreater(len(tags_records, 0))
        for stream in self.expected_sync_streams():
            messages = records.get(stream, {}).get('messages', [])
            if stream == 'tags':
                if self.tags_are_stale:
                    self.refresh_tags(records)
                else:
                    messages = tags_records.get(stream).get('messages')
            if stream in ['tickets', 'groups', 'users']:
                self.assertGreater(len(messages), 100, msg='Stream {} has fewer than 100 records synced'.format(stream))
            for m in messages:
                pk_set = self.expected_pks()[stream]
                for pk in pk_set:
                    self.assertIsNotNone(m.get('data', {}).get(pk), msg='Missing primary-key for message {}'.format(m))
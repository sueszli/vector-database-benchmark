"""Test tap field selection of child streams without its parent."""
import re
from datetime import datetime as dt
from datetime import timedelta
from tap_tester import connections
from tap_tester import menagerie
from tap_tester import runner
from base import HubspotBaseTest
from client import TestClient

class FieldSelectionChildTest(HubspotBaseTest):
    """Test tap field selection of child streams without its parent."""

    @staticmethod
    def name():
        if False:
            i = 10
            return i + 15
        return 'tt_hubspot_child_streams'

    def get_properties(self):
        if False:
            return 10
        return {'start_date': dt.strftime(dt.today() - timedelta(days=2), self.START_DATE_FORMAT)}

    def setUp(self):
        if False:
            while True:
                i = 10
        test_client = TestClient(start_date=self.get_properties()['start_date'])
        contact = test_client.create('contacts')
        company = test_client.create('companies')[0]
        contact_by_company = test_client.create_contacts_by_company(company_ids=[company['companyId']], contact_records=contact)

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Verify that when a child stream is selected without its parent that\n        • a critical error in the tap occurs\n        • the error indicates which parent stream needs to be selected\n        • when the parent is selected the tap doesn't critical error\n        "
        streams_to_test = {'contacts_by_company'}
        conn_id = self.create_connection_and_run_check()
        found_catalogs = self.run_and_verify_check_mode(conn_id)
        catalog_entries = [ce for ce in found_catalogs if ce['tap_stream_id'] in streams_to_test]
        for catalog_entry in catalog_entries:
            stream_schema = menagerie.get_annotated_schema(conn_id, catalog_entry['stream_id'])
            connections.select_catalog_and_fields_via_metadata(conn_id, catalog_entry, stream_schema)
        sync_job_name = runner.run_sync_mode(self, conn_id)
        exit_status = menagerie.get_exit_status(conn_id, sync_job_name)
        self.assertRaises(AssertionError, menagerie.verify_sync_exit_status, self, exit_status, sync_job_name)
        self.assertEqual(exit_status['tap_error_message'], 'Unable to extract contacts_by_company data. To receive contacts_by_company data, you also need to select companies.')
        self.assertEqual(exit_status['target_exit_status'], 0)
        self.assertEqual(exit_status['discovery_exit_status'], 0)
        streams_to_test = {'contacts_by_company', 'companies'}
        catalog_entries = [ce for ce in found_catalogs if ce['tap_stream_id'] in streams_to_test]
        for catalog_entry in catalog_entries:
            stream_schema = menagerie.get_annotated_schema(conn_id, catalog_entry['stream_id'])
            connections.select_catalog_and_fields_via_metadata(conn_id, catalog_entry, stream_schema)
        self.run_and_verify_sync(conn_id)
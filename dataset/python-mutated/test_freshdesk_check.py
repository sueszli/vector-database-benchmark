"""Test tap check mode and metadata/annotated-schema."""
import re
from tap_tester import menagerie, connections, runner
from base import FreshdeskBaseTest

class FreshdeskCheckTest(FreshdeskBaseTest):
    """Test tap check  mode and metadata/annotated-schema conforms to standards."""

    @staticmethod
    def name():
        if False:
            print('Hello World!')
        return 'tt_freshdesk_check'

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Freshdesk check test (does not run discovery).\n        Verify that check does NOT create a discovery catalog, schema, metadata, etc.\n\n        • Verify check job does not populate found_catalogs\n        • Verify no critical errors are thrown for check job\n        '
        streams_to_test = self.expected_streams()
        conn_id = connections.ensure_connection(self)
        self.run_and_verify_check_mode(conn_id)
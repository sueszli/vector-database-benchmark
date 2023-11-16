from frappe.core.report.database_storage_usage_by_tables.database_storage_usage_by_tables import execute
from frappe.tests.utils import FrappeTestCase

class TestDBUsageReport(FrappeTestCase):

    def test_basic_query(self):
        if False:
            return 10
        (_, data) = execute()
        tables = [d.table for d in data]
        self.assertFalse({'tabUser', 'tabDocField'}.difference(tables))
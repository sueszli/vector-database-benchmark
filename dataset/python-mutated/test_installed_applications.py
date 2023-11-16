import frappe
from frappe.core.doctype.installed_applications.installed_applications import InvalidAppOrder, update_installed_apps_order
from frappe.tests.utils import FrappeTestCase

class TestInstalledApplications(FrappeTestCase):

    def test_order_change(self):
        if False:
            return 10
        update_installed_apps_order(['frappe'])
        self.assertRaises(InvalidAppOrder, update_installed_apps_order, [])
        self.assertRaises(InvalidAppOrder, update_installed_apps_order, ['frappe', 'deepmind'])
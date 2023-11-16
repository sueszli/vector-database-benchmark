import frappe
from frappe.contacts.doctype.address_template.address_template import get_default_address_template
from frappe.tests.utils import FrappeTestCase
from frappe.utils.jinja import validate_template

class TestAddressTemplate(FrappeTestCase):

    def setUp(self) -> None:
        if False:
            return 10
        frappe.db.delete('Address Template', {'country': 'India'})
        frappe.db.delete('Address Template', {'country': 'Brazil'})

    def test_default_address_template(self):
        if False:
            for i in range(10):
                print('nop')
        validate_template(get_default_address_template())

    def test_default_is_unset(self):
        if False:
            i = 10
            return i + 15
        frappe.get_doc({'doctype': 'Address Template', 'country': 'India', 'is_default': 1}).insert()
        self.assertEqual(frappe.db.get_value('Address Template', 'India', 'is_default'), 1)
        frappe.get_doc({'doctype': 'Address Template', 'country': 'Brazil', 'is_default': 1}).insert()
        self.assertEqual(frappe.db.get_value('Address Template', 'India', 'is_default'), 0)
        self.assertEqual(frappe.db.get_value('Address Template', 'Brazil', 'is_default'), 1)

    def test_delete_address_template(self):
        if False:
            i = 10
            return i + 15
        india = frappe.get_doc({'doctype': 'Address Template', 'country': 'India', 'is_default': 0}).insert()
        brazil = frappe.get_doc({'doctype': 'Address Template', 'country': 'Brazil', 'is_default': 1}).insert()
        india.reload()
        india.delete()
        self.assertRaises(frappe.ValidationError, brazil.delete)
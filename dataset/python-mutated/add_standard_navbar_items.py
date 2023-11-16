import frappe
from frappe.utils.install import add_standard_navbar_items

def execute():
    if False:
        while True:
            i = 10
    frappe.reload_doc('core', 'doctype', 'navbar_settings')
    frappe.reload_doc('core', 'doctype', 'navbar_item')
    add_standard_navbar_items()
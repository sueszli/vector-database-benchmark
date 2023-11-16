import frappe

def execute():
    if False:
        while True:
            i = 10
    item = frappe.db.exists('Navbar Item', {'item_label': 'Background Jobs'})
    if not item:
        return
    frappe.delete_doc('Navbar Item', item)
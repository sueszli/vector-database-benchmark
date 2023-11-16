import frappe

def execute():
    if False:
        print('Hello World!')
    for name in ('desktop', 'space'):
        frappe.delete_doc('Page', name)
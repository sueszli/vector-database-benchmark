import frappe

def execute():
    if False:
        print('Hello World!')
    frappe.delete_doc('DocType', 'Package Publish Tool', ignore_missing=True)
    frappe.delete_doc('DocType', 'Package Document Type', ignore_missing=True)
    frappe.delete_doc('DocType', 'Package Publish Target', ignore_missing=True)
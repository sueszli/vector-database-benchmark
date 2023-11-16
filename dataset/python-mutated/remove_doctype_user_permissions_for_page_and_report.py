import frappe

def execute():
    if False:
        return 10
    frappe.delete_doc_if_exists('DocType', 'User Permission for Page and Report')
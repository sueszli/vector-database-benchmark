import frappe

def execute():
    if False:
        return 10
    frappe.delete_doc_if_exists('DocType', 'Post')
    frappe.delete_doc_if_exists('DocType', 'Post Comment')
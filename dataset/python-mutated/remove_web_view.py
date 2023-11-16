import frappe

def execute():
    if False:
        while True:
            i = 10
    frappe.delete_doc_if_exists('DocType', 'Web View')
    frappe.delete_doc_if_exists('DocType', 'Web View Component')
    frappe.delete_doc_if_exists('DocType', 'CSS Class')
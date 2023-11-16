import frappe

def execute():
    if False:
        return 10
    frappe.reload_doc('email', 'doctype', 'Newsletter')
    frappe.db.sql("\n\t\tUPDATE tabNewsletter\n\t\tSET content_type = 'Rich Text'\n\t")
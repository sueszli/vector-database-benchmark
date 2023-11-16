import frappe

def execute():
    if False:
        return 10
    frappe.reload_doctype('Letter Head')
    frappe.db.sql("update `tabLetter Head` set source = 'HTML'")
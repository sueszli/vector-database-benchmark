import frappe

def execute():
    if False:
        return 10
    'Enable all the existing Client script'
    frappe.db.sql('\n\t\tUPDATE `tabClient Script` SET enabled=1\n\t')
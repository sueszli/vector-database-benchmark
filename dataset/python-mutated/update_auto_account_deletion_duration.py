import frappe

def execute():
    if False:
        return 10
    days = frappe.db.get_single_value('Website Settings', 'auto_account_deletion')
    frappe.db.set_single_value('Website Settings', 'auto_account_deletion', days * 24)
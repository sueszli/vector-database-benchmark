import frappe

def execute():
    if False:
        return 10
    frappe.reload_doc('core', 'doctype', 'system_settings', force=1)
    frappe.db.set_single_value('System Settings', 'password_reset_limit', 3)
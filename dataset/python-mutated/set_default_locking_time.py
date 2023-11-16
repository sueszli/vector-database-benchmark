import frappe

def execute():
    if False:
        i = 10
        return i + 15
    frappe.reload_doc('core', 'doctype', 'system_settings')
    frappe.db.set_single_value('System Settings', 'allow_login_after_fail', 60)
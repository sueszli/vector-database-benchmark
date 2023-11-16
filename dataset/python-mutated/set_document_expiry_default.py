import frappe

def execute():
    if False:
        for i in range(10):
            print('nop')
    frappe.db.set_single_value('System Settings', {'document_share_key_expiry': 30, 'allow_older_web_view_links': 1})
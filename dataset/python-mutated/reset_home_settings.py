import frappe

def execute():
    if False:
        for i in range(10):
            print('nop')
    frappe.reload_doc('core', 'doctype', 'user')
    frappe.db.sql("\n\t\tUPDATE `tabUser`\n\t\tSET `home_settings` = ''\n\t\tWHERE `user_type` = 'System User'\n\t")
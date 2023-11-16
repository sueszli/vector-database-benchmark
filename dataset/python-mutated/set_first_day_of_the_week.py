import frappe

def execute():
    if False:
        i = 10
        return i + 15
    frappe.reload_doctype('System Settings')
    frappe.db.set_single_value('System Settings', 'first_day_of_the_week', 'Monday')
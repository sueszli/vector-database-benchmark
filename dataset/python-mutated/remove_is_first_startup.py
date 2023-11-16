import frappe

def execute():
    if False:
        i = 10
        return i + 15
    singles = frappe.qb.Table('tabSingles')
    frappe.qb.from_(singles).delete().where((singles.doctype == 'System Settings') & (singles.field == 'is_first_startup')).run()
import frappe

def execute():
    if False:
        while True:
            i = 10
    frappe.reload_doc('core', 'doctype', 'DocField')
    if frappe.db.has_column('DocField', 'show_days'):
        frappe.db.sql('\n\t\t\tUPDATE\n\t\t\t\ttabDocField\n\t\t\tSET\n\t\t\t\thide_days = 1 WHERE show_days = 0\n\t\t')
        frappe.db.sql_ddl('alter table tabDocField drop column show_days')
    if frappe.db.has_column('DocField', 'show_seconds'):
        frappe.db.sql('\n\t\t\tUPDATE\n\t\t\t\ttabDocField\n\t\t\tSET\n\t\t\t\thide_seconds = 1 WHERE show_seconds = 0\n\t\t')
        frappe.db.sql_ddl('alter table tabDocField drop column show_seconds')
    frappe.clear_cache(doctype='DocField')
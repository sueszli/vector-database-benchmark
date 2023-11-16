import frappe

def execute():
    if False:
        i = 10
        return i + 15
    if frappe.db.db_type == 'mariadb':
        frappe.db.sql_ddl('alter table `tabSingles` modify column `value` longtext')
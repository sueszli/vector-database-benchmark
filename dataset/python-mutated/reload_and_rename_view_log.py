import frappe

def execute():
    if False:
        return 10
    if frappe.db.table_exists('View log'):
        frappe.db.sql('CREATE TABLE `ViewLogTemp` AS SELECT * FROM `tabView log`')
        frappe.db.sql('DROP table `tabView log`')
        frappe.delete_doc('DocType', 'View log')
        frappe.reload_doc('core', 'doctype', 'view_log')
        frappe.db.sql('INSERT INTO `tabView Log` SELECT * FROM `ViewLogTemp`')
        frappe.db.commit()
        frappe.db.sql('DROP table `ViewLogTemp`')
    else:
        frappe.reload_doc('core', 'doctype', 'view_log')
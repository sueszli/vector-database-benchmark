import frappe

def execute():
    if False:
        print('Hello World!')
    frappe.db.sql('UPDATE `tabUser Permission`\n\t\tSET `modified`=NOW(), `creation`=NOW()\n\t\tWHERE `creation` IS NULL')
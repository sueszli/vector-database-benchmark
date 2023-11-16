import frappe

def execute():
    if False:
        for i in range(10):
            print('nop')
    frappe.db.change_column_type('__Auth', column='password', type='TEXT')
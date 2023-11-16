import frappe

def execute():
    if False:
        for i in range(10):
            print('nop')
    frappe.db.delete('DocType', {'name': 'Feedback Request'})
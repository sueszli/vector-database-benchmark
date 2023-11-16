import frappe

def execute():
    if False:
        return 10
    'Remove stale docfields from legacy version'
    frappe.db.delete('DocField', {'options': 'Data Import', 'parent': 'Data Import Legacy'})
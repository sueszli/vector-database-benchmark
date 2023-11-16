"""
Run this after updating country_info.json and or
"""
import frappe

def execute():
    if False:
        i = 10
        return i + 15
    for col in ('field', 'doctype'):
        frappe.db.sql_ddl(f'alter table `tabSingles` modify column `{col}` varchar(255)')
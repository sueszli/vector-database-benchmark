import frappe

def execute():
    if False:
        i = 10
        return i + 15
    if frappe.db.exists('DocType', 'Onboarding'):
        frappe.rename_doc('DocType', 'Onboarding', 'Module Onboarding', ignore_if_exists=True)
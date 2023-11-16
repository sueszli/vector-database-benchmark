import frappe

def execute():
    if False:
        return 10
    frappe.db.set_value('Currency', 'USD', 'smallest_currency_fraction_value', '0.01')
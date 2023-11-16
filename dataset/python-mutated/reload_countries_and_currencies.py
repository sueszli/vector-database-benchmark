"""
Run this after updating country_info.json and or
"""
from frappe.utils.install import import_country_and_currency

def execute():
    if False:
        while True:
            i = 10
    import_country_and_currency()
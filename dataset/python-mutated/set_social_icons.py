import frappe

def execute():
    if False:
        i = 10
        return i + 15
    providers = frappe.get_all('Social Login Key')
    for provider in providers:
        doc = frappe.get_doc('Social Login Key', provider)
        doc.set_icon()
        doc.save()
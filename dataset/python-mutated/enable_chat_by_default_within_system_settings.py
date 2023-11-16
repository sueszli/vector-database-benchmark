import frappe

def execute():
    if False:
        while True:
            i = 10
    frappe.reload_doctype('System Settings')
    doc = frappe.get_single('System Settings')
    doc.enable_chat = 1
    doc.flags.ignore_mandatory = True
    doc.flags.ignore_permissions = True
    doc.save()
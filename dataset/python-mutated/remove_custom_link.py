import frappe

def execute():
    if False:
        print('Hello World!')
    '\n\tRemove the doctype "Custom Link" that was used to add Custom Links to the\n\tDashboard since this is now managed by Customize Form.\n\tUpdate `parent` property to the DocType and delte the doctype\n\t'
    frappe.reload_doctype('DocType Link')
    if frappe.db.has_table('Custom Link'):
        for custom_link in frappe.get_all('Custom Link', ['name', 'document_type']):
            frappe.db.sql('update `tabDocType Link` set custom=1, parent=%s where parent=%s', (custom_link.document_type, custom_link.name))
        frappe.delete_doc('DocType', 'Custom Link')
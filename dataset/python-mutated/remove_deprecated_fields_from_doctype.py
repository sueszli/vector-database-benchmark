import frappe

def execute():
    if False:
        for i in range(10):
            print('nop')
    frappe.reload_doc('core', 'doctype', 'doctype_link')
    frappe.reload_doc('core', 'doctype', 'doctype_action')
    frappe.reload_doc('core', 'doctype', 'doctype')
    frappe.model.delete_fields({'DocType': ['hide_heading', 'image_view', 'read_only_onload']}, delete=1)
    frappe.db.delete('Property Setter', {'property': 'read_only_onload'})
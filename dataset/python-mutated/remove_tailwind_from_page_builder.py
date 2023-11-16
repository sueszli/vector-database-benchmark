import frappe

def execute():
    if False:
        print('Hello World!')
    frappe.reload_doc('website', 'doctype', 'web_page_block')
    frappe.delete_doc('Web Template', 'Navbar with Links on Right', force=1)
    frappe.delete_doc('Web Template', 'Footer Horizontal', force=1)
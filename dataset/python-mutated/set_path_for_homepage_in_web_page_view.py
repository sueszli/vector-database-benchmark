import frappe

def execute():
    if False:
        print('Hello World!')
    frappe.reload_doc('website', 'doctype', 'web_page_view', force=True)
    frappe.db.sql("UPDATE `tabWeb Page View` set path='/' where path=''")
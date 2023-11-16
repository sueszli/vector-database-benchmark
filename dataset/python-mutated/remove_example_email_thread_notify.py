import frappe

def execute():
    if False:
        print('Hello World!')
    frappe.db.sql("UPDATE `tabUser`\n\tSET thread_notify=0, send_me_a_copy=0\n\tWHERE email like '%@example.com'")
import frappe

def execute():
    if False:
        while True:
            i = 10
    frappe.db.sql("\n\t\tUPDATE tabFile\n\t\tSET folder = 'Home/Attachments'\n\t\tWHERE ifnull(attached_to_doctype, '') != ''\n\t\tAND folder = 'Home'\n\t")
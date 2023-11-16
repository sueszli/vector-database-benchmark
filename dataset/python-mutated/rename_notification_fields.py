import frappe
from frappe.model.utils.rename_field import rename_field

def execute():
    if False:
        i = 10
        return i + 15
    '\n\tChange notification recipient fields from email to receiver fields\n\t'
    frappe.reload_doc('Email', 'doctype', 'Notification Recipient')
    frappe.reload_doc('Email', 'doctype', 'Notification')
    rename_field('Notification Recipient', 'email_by_document_field', 'receiver_by_document_field')
    rename_field('Notification Recipient', 'email_by_role', 'receiver_by_role')
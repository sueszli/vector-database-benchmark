import click
import frappe

def execute():
    if False:
        for i in range(10):
            print('nop')
    frappe.delete_doc_if_exists('DocType', 'Chat Message')
    frappe.delete_doc_if_exists('DocType', 'Chat Message Attachment')
    frappe.delete_doc_if_exists('DocType', 'Chat Profile')
    frappe.delete_doc_if_exists('DocType', 'Chat Token')
    frappe.delete_doc_if_exists('DocType', 'Chat Room User')
    frappe.delete_doc_if_exists('DocType', 'Chat Room')
    frappe.delete_doc_if_exists('Module Def', 'Chat')
    click.secho('Chat Module is moved to a separate app and is removed from Frappe in version-13.\nPlease install the app to continue using the chat feature: https://github.com/frappe/chat', fg='yellow')
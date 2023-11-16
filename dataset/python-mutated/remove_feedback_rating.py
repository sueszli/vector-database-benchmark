import frappe

def execute():
    if False:
        i = 10
        return i + 15
    '\n\tDeprecate Feedback Trigger and Rating. This feature was not customizable.\n\tNow can be achieved via custom Web Forms\n\t'
    frappe.delete_doc('DocType', 'Feedback Trigger')
    frappe.delete_doc('DocType', 'Feedback Rating')
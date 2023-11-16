import frappe

def execute():
    if False:
        i = 10
        return i + 15
    'Add missing Twilio patch.\n\n\tWhile making Twilio as a standaone app, we missed to delete Twilio records from DB through migration. Adding the missing patch.\n\t'
    frappe.delete_doc_if_exists('DocType', 'Twilio Number Group')
    if twilio_settings_doctype_in_integrations():
        frappe.delete_doc_if_exists('DocType', 'Twilio Settings')
        frappe.db.delete('Singles', {'doctype': 'Twilio Settings'})

def twilio_settings_doctype_in_integrations() -> bool:
    if False:
        print('Hello World!')
    'Check Twilio Settings doctype exists in integrations module or not.'
    return frappe.db.exists('DocType', {'name': 'Twilio Settings', 'module': 'Integrations'})
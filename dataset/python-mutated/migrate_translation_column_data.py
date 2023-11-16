import frappe

def execute():
    if False:
        return 10
    frappe.reload_doctype('Translation')
    frappe.db.sql('UPDATE `tabTranslation` SET `translated_text`=`target_name`, `source_text`=`source_name`, `contributed`=0')
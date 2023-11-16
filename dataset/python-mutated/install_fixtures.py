import frappe
from frappe.desk.doctype.global_search_settings.global_search_settings import update_global_search_doctypes
from frappe.utils.dashboard import sync_dashboards

def install():
    if False:
        while True:
            i = 10
    update_genders()
    update_salutations()
    update_global_search_doctypes()
    setup_email_linking()
    sync_dashboards()
    add_unsubscribe()

def update_genders():
    if False:
        for i in range(10):
            print('nop')
    default_genders = ['Male', 'Female', 'Other', 'Transgender', 'Genderqueer', 'Non-Conforming', 'Prefer not to say']
    records = [{'doctype': 'Gender', 'gender': d} for d in default_genders]
    for record in records:
        frappe.get_doc(record).insert(ignore_permissions=True, ignore_if_duplicate=True)

def update_salutations():
    if False:
        return 10
    default_salutations = ['Mr', 'Ms', 'Mx', 'Dr', 'Mrs', 'Madam', 'Miss', 'Master', 'Prof']
    records = [{'doctype': 'Salutation', 'salutation': d} for d in default_salutations]
    for record in records:
        doc = frappe.new_doc(record.get('doctype'))
        doc.update(record)
        doc.insert(ignore_permissions=True, ignore_if_duplicate=True)

def setup_email_linking():
    if False:
        i = 10
        return i + 15
    doc = frappe.get_doc({'doctype': 'Email Account', 'email_id': 'email_linking@example.com'})
    doc.insert(ignore_permissions=True, ignore_if_duplicate=True)

def add_unsubscribe():
    if False:
        i = 10
        return i + 15
    email_unsubscribe = [{'email': 'admin@example.com', 'global_unsubscribe': 1}, {'email': 'guest@example.com', 'global_unsubscribe': 1}]
    for unsubscribe in email_unsubscribe:
        if not frappe.get_all('Email Unsubscribe', filters=unsubscribe):
            doc = frappe.new_doc('Email Unsubscribe')
            doc.update(unsubscribe)
            doc.insert(ignore_permissions=True)
import frappe

def execute():
    if False:
        print('Hello World!')
    for email_account in frappe.get_list('Email Account', filters={'enable_incoming': 1, 'use_imap': 1}):
        doc = frappe.get_doc('Email Account', email_account.name)
        imap_list = [folder.folder_name for folder in doc.imap_folder]
        if doc.uidvalidity or (doc.uidnext and 'INBOX' not in imap_list):
            doc.append('imap_folder', {'folder_name': 'INBOX', 'append_to': doc.append_to, 'uid_validity': doc.uidvalidity, 'uidnext': doc.uidnext})
            doc.save()
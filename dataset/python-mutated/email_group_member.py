import frappe
from frappe.model.document import Document

class EmailGroupMember(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        email: DF.Data
        email_group: DF.Link
        unsubscribed: DF.Check

    def after_delete(self):
        if False:
            while True:
                i = 10
        email_group = frappe.get_doc('Email Group', self.email_group)
        email_group.update_total_subscribers()

    def after_insert(self):
        if False:
            for i in range(10):
                print('nop')
        email_group = frappe.get_doc('Email Group', self.email_group)
        email_group.update_total_subscribers()

def after_doctype_insert():
    if False:
        while True:
            i = 10
    frappe.db.add_unique('Email Group Member', ('email_group', 'email'))
import frappe
from frappe.model.document import Document

class EmailQueueRecipient(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        error: DF.Code | None
        parent: DF.Data
        parentfield: DF.Data
        parenttype: DF.Data
        recipient: DF.Data | None
        status: DF.Literal['', 'Not Sent', 'Sent']
    DOCTYPE = 'Email Queue Recipient'

    def is_mail_to_be_sent(self):
        if False:
            while True:
                i = 10
        return self.status == 'Not Sent'

    def is_mail_sent(self):
        if False:
            return 10
        return self.status == 'Sent'

    def update_db(self, commit=False, **kwargs):
        if False:
            print('Hello World!')
        frappe.db.set_value(self.DOCTYPE, self.name, kwargs)
        if commit:
            frappe.db.commit()

def on_doctype_update():
    if False:
        return 10
    'Index required for log clearing, modified is not indexed on child table by default'
    frappe.db.add_index('Email Queue Recipient', ['modified'])
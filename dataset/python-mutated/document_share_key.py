from random import randrange
import frappe
from frappe.model.document import Document

class DocumentShareKey(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        expires_on: DF.Date | None
        key: DF.Data | None
        reference_docname: DF.DynamicLink | None
        reference_doctype: DF.Link | None

    def before_insert(self):
        if False:
            print('Hello World!')
        self.key = frappe.generate_hash(length=randrange(25, 35))
        if not self.expires_on and (not self.flags.no_expiry):
            self.expires_on = frappe.utils.add_days(None, days=frappe.get_system_settings('document_share_key_expiry') or 90)

def is_expired(expires_on):
    if False:
        return 10
    return expires_on and expires_on < frappe.utils.getdate()
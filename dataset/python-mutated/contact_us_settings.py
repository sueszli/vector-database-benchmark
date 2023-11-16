import frappe
from frappe.model.document import Document

class ContactUsSettings(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        address_line1: DF.Data | None
        address_line2: DF.Data | None
        address_title: DF.Data | None
        city: DF.Data | None
        country: DF.Data | None
        disable_contact_us: DF.Check
        email_id: DF.Data | None
        forward_to_email: DF.Data | None
        heading: DF.Data | None
        introduction: DF.TextEditor | None
        phone: DF.Data | None
        pincode: DF.Data | None
        query_options: DF.SmallText | None
        skype: DF.Data | None
        state: DF.Data | None

    def on_update(self):
        if False:
            i = 10
            return i + 15
        from frappe.website.utils import clear_cache
        clear_cache('contact')
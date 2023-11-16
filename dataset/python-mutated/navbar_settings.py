import frappe
from frappe import _
from frappe.model.document import Document

class NavbarSettings(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.core.doctype.navbar_item.navbar_item import NavbarItem
        from frappe.types import DF
        app_logo: DF.AttachImage | None
        help_dropdown: DF.Table[NavbarItem]
        logo_width: DF.Int
        settings_dropdown: DF.Table[NavbarItem]

    def validate(self):
        if False:
            print('Hello World!')
        self.validate_standard_navbar_items()

    def validate_standard_navbar_items(self):
        if False:
            while True:
                i = 10
        doc_before_save = self.get_doc_before_save()
        if not doc_before_save:
            return
        before_save_items = [item for item in doc_before_save.help_dropdown + doc_before_save.settings_dropdown if item.is_standard]
        after_save_items = [item for item in self.help_dropdown + self.settings_dropdown if item.is_standard]
        if not frappe.flags.in_patch and len(before_save_items) > len(after_save_items):
            frappe.throw(_('Please hide the standard navbar items instead of deleting them'))

def get_app_logo():
    if False:
        while True:
            i = 10
    app_logo = frappe.db.get_single_value('Navbar Settings', 'app_logo', cache=True)
    if not app_logo:
        app_logo = frappe.get_hooks('app_logo_url')[-1]
    return app_logo

def get_navbar_settings():
    if False:
        for i in range(10):
            print('nop')
    return frappe.get_single('Navbar Settings')
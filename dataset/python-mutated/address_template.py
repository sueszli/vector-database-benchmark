import frappe
from frappe import _
from frappe.model.document import Document
from frappe.utils.jinja import validate_template

class AddressTemplate(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        country: DF.Link
        is_default: DF.Check
        template: DF.Code | None

    def validate(self):
        if False:
            while True:
                i = 10
        validate_template(self.template)
        if not self.template:
            self.template = get_default_address_template()
        if not self.is_default and (not self._get_previous_default()):
            self.is_default = 1
            if frappe.db.get_single_value('System Settings', 'setup_complete'):
                frappe.msgprint(_('Setting this Address Template as default as there is no other default'))

    def on_update(self):
        if False:
            print('Hello World!')
        if self.is_default and (previous_default := self._get_previous_default()):
            frappe.db.set_value('Address Template', previous_default, 'is_default', 0)

    def on_trash(self):
        if False:
            return 10
        if self.is_default:
            frappe.throw(_('Default Address Template cannot be deleted'))

    def _get_previous_default(self) -> str | None:
        if False:
            i = 10
            return i + 15
        return frappe.db.get_value('Address Template', {'is_default': 1, 'name': ('!=', self.name)})

@frappe.whitelist()
def get_default_address_template() -> str:
    if False:
        for i in range(10):
            print('nop')
    'Return the default address template.'
    from pathlib import Path
    return (Path(__file__).parent / 'address_template.jinja').read_text()
import frappe
from frappe.model.document import Document

class PrintStyle(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        css: DF.Code
        disabled: DF.Check
        preview: DF.AttachImage | None
        print_style_name: DF.Data
        standard: DF.Check

    def validate(self):
        if False:
            return 10
        if self.standard == 1 and (not frappe.local.conf.get('developer_mode')) and (not (frappe.flags.in_import or frappe.flags.in_test)):
            frappe.throw(frappe._('Standard Print Style cannot be changed. Please duplicate to edit.'))

    def on_update(self):
        if False:
            print('Hello World!')
        self.export_doc()

    def export_doc(self):
        if False:
            print('Hello World!')
        from frappe.modules.utils import export_module_json
        export_module_json(self, self.standard == 1, 'Printing')
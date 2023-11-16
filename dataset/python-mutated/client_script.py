import frappe
from frappe.model.document import Document

class ClientScript(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        dt: DF.Link
        enabled: DF.Check
        module: DF.Link | None
        script: DF.Code | None
        view: DF.Literal['List', 'Form']

    def on_update(self):
        if False:
            print('Hello World!')
        frappe.clear_cache(doctype=self.dt)

    def on_trash(self):
        if False:
            i = 10
            return i + 15
        frappe.clear_cache(doctype=self.dt)
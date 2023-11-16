import frappe
from frappe.model.document import Document

class CommunicationLink(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        link_doctype: DF.Link
        link_name: DF.DynamicLink
        link_title: DF.ReadOnly | None
        parent: DF.Data
        parentfield: DF.Data
        parenttype: DF.Data
    pass

def on_doctype_update():
    if False:
        print('Hello World!')
    frappe.db.add_index('Communication Link', ['link_doctype', 'link_name'])
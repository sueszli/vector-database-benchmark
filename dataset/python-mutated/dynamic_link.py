import frappe
from frappe.model.document import Document

class DynamicLink(Document):
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
    frappe.db.add_index('Dynamic Link', ['link_doctype', 'link_name'])

def deduplicate_dynamic_links(doc):
    if False:
        for i in range(10):
            print('nop')
    (links, duplicate) = ([], False)
    for l in doc.links or []:
        t = (l.link_doctype, l.link_name)
        if not t in links:
            links.append(t)
        else:
            duplicate = True
    if duplicate:
        doc.links = []
        for l in links:
            doc.append('links', dict(link_doctype=l[0], link_name=l[1]))
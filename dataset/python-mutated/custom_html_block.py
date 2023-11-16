import frappe
from frappe.model.document import Document
from frappe.query_builder.utils import DocType

class CustomHTMLBlock(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.core.doctype.has_role.has_role import HasRole
        from frappe.types import DF
        html: DF.Code | None
        private: DF.Check
        roles: DF.Table[HasRole]
        script: DF.Code | None
        style: DF.Code | None
    pass

@frappe.whitelist()
def get_custom_blocks_for_user(doctype, txt, searchfield, start, page_len, filters):
    if False:
        while True:
            i = 10
    customHTMLBlock = DocType('Custom HTML Block')
    condition_query = frappe.qb.from_(customHTMLBlock)
    return condition_query.select(customHTMLBlock.name).where((customHTMLBlock.private == 0) | (customHTMLBlock.owner == frappe.session.user) & (customHTMLBlock.private == 1)).run()
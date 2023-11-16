import frappe
from frappe.model.document import Document

class CustomDocPerm(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        amend: DF.Check
        cancel: DF.Check
        create: DF.Check
        delete: DF.Check
        email: DF.Check
        export: DF.Check
        if_owner: DF.Check
        parent: DF.Data | None
        permlevel: DF.Int
        print: DF.Check
        read: DF.Check
        report: DF.Check
        role: DF.Link
        select: DF.Check
        share: DF.Check
        submit: DF.Check
        write: DF.Check

    def on_update(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.clear_cache(doctype=self.parent)
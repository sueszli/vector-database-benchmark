import frappe
from frappe.model.document import Document

class PatchLog(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        patch: DF.Code | None
        skipped: DF.Check
        traceback: DF.Code | None
    pass

def before_migrate():
    if False:
        for i in range(10):
            print('nop')
    frappe.reload_doc('core', 'doctype', 'patch_log')
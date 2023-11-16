import frappe
from frappe.model.document import Document

class HasRole(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        parent: DF.Data
        parentfield: DF.Data
        parenttype: DF.Data
        role: DF.Link | None

    def before_insert(self):
        if False:
            i = 10
            return i + 15
        if frappe.db.exists('Has Role', {'parent': self.parent, 'role': self.role}):
            frappe.throw(frappe._("User '{0}' already has the role '{1}'").format(self.parent, self.role))
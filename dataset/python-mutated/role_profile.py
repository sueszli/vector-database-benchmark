import frappe
from frappe.model.document import Document

class RoleProfile(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.core.doctype.has_role.has_role import HasRole
        from frappe.types import DF
        role_profile: DF.Data
        roles: DF.Table[HasRole]

    def autoname(self):
        if False:
            print('Hello World!')
        'set name as Role Profile name'
        self.name = self.role_profile

    def on_update(self):
        if False:
            while True:
                i = 10
        'Changes in role_profile reflected across all its user'
        users = frappe.get_all('User', filters={'role_profile_name': self.name})
        roles = [role.role for role in self.roles]
        for d in users:
            user = frappe.get_doc('User', d)
            user.set('roles', [])
            user.add_roles(*roles)
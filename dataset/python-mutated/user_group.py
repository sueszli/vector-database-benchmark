import frappe
from frappe.model.document import Document

class UserGroup(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.core.doctype.user_group_member.user_group_member import UserGroupMember
        from frappe.types import DF
        user_group_members: DF.TableMultiSelect[UserGroupMember]

    def after_insert(self):
        if False:
            print('Hello World!')
        frappe.cache.delete_key('user_groups')

    def on_trash(self):
        if False:
            return 10
        frappe.cache.delete_key('user_groups')
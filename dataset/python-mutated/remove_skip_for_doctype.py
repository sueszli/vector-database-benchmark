import frappe
from frappe.desk.form.linked_with import get_linked_doctypes
from frappe.patches.v11_0.replicate_old_user_permissions import get_doctypes_to_skip
from frappe.query_builder import Field

def execute():
    if False:
        print('Hello World!')
    frappe.reload_doctype('User Permission')
    has_skip_for_doctype = frappe.db.has_column('User Permission', 'skip_for_doctype')
    skip_for_doctype_map = {}
    new_user_permissions_list = []
    user_permissions_to_delete = []
    for user_permission in frappe.get_all('User Permission', fields=['*']):
        skip_for_doctype = []
        if has_skip_for_doctype:
            if not user_permission.skip_for_doctype:
                continue
            skip_for_doctype = user_permission.skip_for_doctype.split('\n')
        elif skip_for_doctype_map.get((user_permission.allow, user_permission.user)) is None:
            skip_for_doctype = get_doctypes_to_skip(user_permission.allow, user_permission.user)
            skip_for_doctype_map[user_permission.allow, user_permission.user] = skip_for_doctype
        else:
            skip_for_doctype = skip_for_doctype_map[user_permission.allow, user_permission.user]
        if skip_for_doctype:
            linked_doctypes = get_linked_doctypes(user_permission.allow, True).keys()
            linked_doctypes = list(linked_doctypes)
            linked_doctypes += [user_permission.allow]
            applicable_for_doctypes = list(set(linked_doctypes) - set(skip_for_doctype))
            user_permissions_to_delete.append(user_permission.name)
            user_permission.name = None
            user_permission.skip_for_doctype = None
            new_user_permissions_list.extend(((frappe.generate_hash(length=10), user_permission.user, user_permission.allow, user_permission.for_value, doctype, 0, user_permission.creation, user_permission.modified) for doctype in applicable_for_doctypes if doctype))
        else:
            frappe.db.set_value('User Permission', user_permission.name, 'apply_to_all_doctypes', 1)
    if new_user_permissions_list:
        frappe.qb.into('User Permission').columns('name', 'user', 'allow', 'for_value', 'applicable_for', 'apply_to_all_doctypes', 'creation', 'modified').insert(*new_user_permissions_list).run()
    if user_permissions_to_delete:
        frappe.db.delete('User Permission', filters=Field('name').isin(tuple(user_permissions_to_delete)))
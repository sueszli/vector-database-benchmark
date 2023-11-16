"""
Create a new document with defaults set
"""
import copy
import frappe
import frappe.defaults
from frappe.core.doctype.user_permission.user_permission import get_user_permissions
from frappe.model import data_fieldtypes
from frappe.permissions import filter_allowed_docs_for_doctype
from frappe.utils import cstr, now_datetime, nowdate, nowtime

def get_new_doc(doctype, parent_doc=None, parentfield=None, as_dict=False):
    if False:
        i = 10
        return i + 15
    if doctype not in frappe.local.new_doc_templates:
        frappe.local.new_doc_templates[doctype] = make_new_doc(doctype)
    doc = copy.deepcopy(frappe.local.new_doc_templates[doctype])
    set_dynamic_default_values(doc, parent_doc, parentfield)
    if as_dict:
        return doc
    else:
        return frappe.get_doc(doc)

def make_new_doc(doctype):
    if False:
        print('Hello World!')
    doc = frappe.get_doc({'doctype': doctype, '__islocal': 1, 'owner': frappe.session.user, 'docstatus': 0})
    set_user_and_static_default_values(doc)
    doc._fix_numeric_types()
    doc = doc.get_valid_dict(sanitize=False)
    doc['doctype'] = doctype
    doc['__islocal'] = 1
    if not frappe.model.meta.is_single(doctype):
        doc['__unsaved'] = 1
    return doc

def set_user_and_static_default_values(doc):
    if False:
        print('Hello World!')
    user_permissions = get_user_permissions()
    defaults = frappe.defaults.get_defaults()
    for df in doc.meta.get('fields'):
        if df.fieldtype in data_fieldtypes:
            doctype_user_permissions = user_permissions.get(df.options, [])
            (allowed_records, default_doc) = filter_allowed_docs_for_doctype(doctype_user_permissions, df.parent, with_default_doc=True)
            user_default_value = get_user_default_value(df, defaults, doctype_user_permissions, allowed_records, default_doc)
            if user_default_value is not None:
                if not df.fieldtype == 'Link' or frappe.db.exists(df.options, user_default_value):
                    doc.set(df.fieldname, user_default_value)
            elif df.fieldname != doc.meta.title_field:
                static_default_value = get_static_default_value(df, doctype_user_permissions, allowed_records)
                if static_default_value is not None:
                    doc.set(df.fieldname, static_default_value)

def get_user_default_value(df, defaults, doctype_user_permissions, allowed_records, default_doc):
    if False:
        while True:
            i = 10
    if df.fieldtype == 'Link' and df.options != 'User':
        if not df.ignore_user_permissions and default_doc:
            return default_doc
        user_default = defaults.get(df.fieldname)
        allowed_by_user_permission = validate_value_via_user_permissions(df, doctype_user_permissions, allowed_records, user_default=user_default)
        if user_default and allowed_by_user_permission:
            return user_default

def get_static_default_value(df, doctype_user_permissions, allowed_records):
    if False:
        print('Hello World!')
    if df.get('default'):
        if df.default == '__user':
            return frappe.session.user
        elif df.default == 'Today':
            return nowdate()
        elif not cstr(df.default).startswith(':'):
            is_allowed_default_value = validate_value_via_user_permissions(df, doctype_user_permissions, allowed_records)
            if df.fieldtype != 'Link' or df.options == 'User' or is_allowed_default_value:
                return df.default
    elif df.fieldtype == 'Select' and df.options and (df.options not in ('[Select]', 'Loading...')):
        return df.options.split('\n', 1)[0]

def validate_value_via_user_permissions(df, doctype_user_permissions, allowed_records, user_default=None):
    if False:
        while True:
            i = 10
    is_valid = True
    if user_permissions_exist(df, doctype_user_permissions) and allowed_records:
        value = user_default if user_default else df.default
        is_valid = value in allowed_records
    return is_valid

def set_dynamic_default_values(doc, parent_doc, parentfield):
    if False:
        for i in range(10):
            print('nop')
    user_permissions = get_user_permissions()
    for df in frappe.get_meta(doc['doctype']).get('fields'):
        if df.get('default'):
            if cstr(df.default).startswith(':'):
                default_value = get_default_based_on_another_field(df, user_permissions, parent_doc)
                if default_value is not None and (not doc.get(df.fieldname)):
                    doc[df.fieldname] = default_value
            elif df.fieldtype == 'Datetime' and df.default.lower() == 'now':
                doc[df.fieldname] = now_datetime()
        if df.fieldtype == 'Time':
            doc[df.fieldname] = nowtime()
    if parent_doc:
        doc['parent'] = parent_doc.name
        doc['parenttype'] = parent_doc.doctype
    if parentfield:
        doc['parentfield'] = parentfield

def user_permissions_exist(df, doctype_user_permissions):
    if False:
        print('Hello World!')
    return df.fieldtype == 'Link' and (not getattr(df, 'ignore_user_permissions', False)) and doctype_user_permissions

def get_default_based_on_another_field(df, user_permissions, parent_doc):
    if False:
        while True:
            i = 10
    from frappe.permissions import get_allowed_docs_for_doctype
    ref_doctype = df.default[1:]
    ref_fieldname = ref_doctype.lower().replace(' ', '_')
    reference_name = parent_doc.get(ref_fieldname) if parent_doc else frappe.db.get_default(ref_fieldname)
    default_value = frappe.db.get_value(ref_doctype, reference_name, df.fieldname)
    is_allowed_default_value = not user_permissions_exist(df, user_permissions.get(df.options)) or default_value in get_allowed_docs_for_doctype(user_permissions[df.options], df.parent)
    if is_allowed_default_value:
        return default_value
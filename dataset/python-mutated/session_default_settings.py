import json
import frappe
from frappe import _
from frappe.model.document import Document

class SessionDefaultSettings(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.core.doctype.session_default.session_default import SessionDefault
        from frappe.types import DF
        session_defaults: DF.Table[SessionDefault]
    pass

@frappe.whitelist()
def get_session_default_values():
    if False:
        print('Hello World!')
    settings = frappe.get_single('Session Default Settings')
    fields = []
    for default_values in settings.session_defaults:
        reference_doctype = frappe.scrub(default_values.ref_doctype)
        fields.append({'fieldname': reference_doctype, 'fieldtype': 'Link', 'options': default_values.ref_doctype, 'label': _('Default {0}').format(_(default_values.ref_doctype)), 'default': frappe.defaults.get_user_default(reference_doctype)})
    return json.dumps(fields)

@frappe.whitelist()
def set_session_default_values(default_values):
    if False:
        for i in range(10):
            print('nop')
    default_values = frappe.parse_json(default_values)
    for entry in default_values:
        try:
            frappe.defaults.set_user_default(entry, default_values.get(entry))
        except Exception:
            return
    return 'success'

def clear_session_defaults():
    if False:
        i = 10
        return i + 15
    settings = frappe.get_single('Session Default Settings').session_defaults
    for entry in settings:
        frappe.defaults.clear_user_default(frappe.scrub(entry.ref_doctype))
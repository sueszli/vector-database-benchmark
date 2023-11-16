import frappe
import frappe.cache_manager
from frappe.model import log_types
from frappe.model.document import Document

class MilestoneTracker(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        disabled: DF.Check
        document_type: DF.Link
        track_field: DF.Literal

    def on_update(self):
        if False:
            print('Hello World!')
        frappe.cache_manager.clear_doctype_map('Milestone Tracker', self.document_type)

    def on_trash(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.cache_manager.clear_doctype_map('Milestone Tracker', self.document_type)

    def apply(self, doc):
        if False:
            print('Hello World!')
        before_save = doc.get_doc_before_save()
        from_value = before_save and before_save.get(self.track_field) or None
        if from_value != doc.get(self.track_field):
            frappe.get_doc(dict(doctype='Milestone', reference_type=doc.doctype, reference_name=doc.name, track_field=self.track_field, from_value=from_value, value=doc.get(self.track_field), milestone_tracker=self.name)).insert(ignore_permissions=True)

def evaluate_milestone(doc, event):
    if False:
        print('Hello World!')
    if frappe.flags.in_install or frappe.flags.in_migrate or frappe.flags.in_setup_wizard or (doc.doctype in log_types):
        return
    for d in get_milestone_trackers(doc.doctype):
        frappe.get_doc('Milestone Tracker', d.get('name')).apply(doc)

def get_milestone_trackers(doctype):
    if False:
        print('Hello World!')
    return frappe.cache_manager.get_doctype_map('Milestone Tracker', doctype, dict(document_type=doctype, disabled=0))
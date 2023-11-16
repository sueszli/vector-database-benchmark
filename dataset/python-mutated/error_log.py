import frappe
from frappe.model.document import Document
from frappe.query_builder import Interval
from frappe.query_builder.functions import Now

class ErrorLog(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        error: DF.Code | None
        method: DF.Data | None
        reference_doctype: DF.Link | None
        reference_name: DF.Data | None
        seen: DF.Check
        trace_id: DF.Data | None

    def onload(self):
        if False:
            return 10
        if not self.seen and (not frappe.flags.read_only):
            self.db_set('seen', 1, update_modified=0)
            frappe.db.commit()

    @staticmethod
    def clear_old_logs(days=30):
        if False:
            for i in range(10):
                print('nop')
        table = frappe.qb.DocType('Error Log')
        frappe.db.delete(table, filters=table.modified < Now() - Interval(days=days))

@frappe.whitelist()
def clear_error_logs():
    if False:
        print('Hello World!')
    'Flush all Error Logs'
    frappe.only_for('System Manager')
    frappe.db.truncate('Error Log')
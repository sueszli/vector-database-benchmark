import frappe
from frappe.model.document import Document

class ViewLog(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        reference_doctype: DF.Link | None
        reference_name: DF.DynamicLink | None
        viewed_by: DF.Data | None

    @staticmethod
    def clear_old_logs(days=180):
        if False:
            return 10
        from frappe.query_builder import Interval
        from frappe.query_builder.functions import Now
        table = frappe.qb.DocType('View Log')
        frappe.db.delete(table, filters=table.modified < Now() - Interval(days=days))
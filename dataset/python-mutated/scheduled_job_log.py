import frappe
from frappe.model.document import Document
from frappe.query_builder import Interval
from frappe.query_builder.functions import Now

class ScheduledJobLog(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        details: DF.Code | None
        scheduled_job_type: DF.Link
        status: DF.Literal['Scheduled', 'Complete', 'Failed']

    @staticmethod
    def clear_old_logs(days=90):
        if False:
            i = 10
            return i + 15
        table = frappe.qb.DocType('Scheduled Job Log')
        frappe.db.delete(table, filters=table.modified < Now() - Interval(days=days))
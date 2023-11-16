import frappe
from frappe.model.document import Document

class WebhookRequestLog(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        data: DF.Code | None
        error: DF.Text | None
        headers: DF.Code | None
        reference_document: DF.Data | None
        response: DF.Code | None
        url: DF.Data | None
        user: DF.Link | None
        webhook: DF.Link | None

    @staticmethod
    def clear_old_logs(days=30):
        if False:
            for i in range(10):
                print('nop')
        from frappe.query_builder import Interval
        from frappe.query_builder.functions import Now
        table = frappe.qb.DocType('Webhook Request Log')
        frappe.db.delete(table, filters=table.modified < Now() - Interval(days=days))
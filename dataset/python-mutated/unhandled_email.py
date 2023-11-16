import frappe
from frappe.model.document import Document

class UnhandledEmail(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        email_account: DF.Link | None
        message_id: DF.Code | None
        raw: DF.Code | None
        reason: DF.LongText | None
        uid: DF.Data | None

    @staticmethod
    def clear_old_logs(days=30):
        if False:
            for i in range(10):
                print('nop')
        frappe.db.delete('Unhandled Email', {'modified': ('<', frappe.utils.add_days(frappe.utils.nowdate(), -1 * days))})
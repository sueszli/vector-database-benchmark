import frappe
from frappe import _
from frappe.model.document import Document
from frappe.utils import cint
from frappe.utils.data import add_to_date, get_datetime, now_datetime

class Reminder(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        description: DF.SmallText
        notified: DF.Check
        remind_at: DF.Datetime
        reminder_docname: DF.DynamicLink | None
        reminder_doctype: DF.Link | None
        user: DF.Link

    @staticmethod
    def clear_old_logs(days=30):
        if False:
            i = 10
            return i + 15
        from frappe.query_builder import Interval
        from frappe.query_builder.functions import Now
        table = frappe.qb.DocType('Reminder')
        frappe.db.delete(table, filters=table.remind_at < Now() - Interval(days=days))

    def validate(self):
        if False:
            return 10
        self.user = frappe.session.user
        if get_datetime(self.remind_at) < now_datetime():
            frappe.throw(_('Reminder cannot be created in past.'))

    def send_reminder(self):
        if False:
            i = 10
            return i + 15
        if self.notified:
            return
        self.db_set('notified', 1, update_modified=False)
        try:
            notification = frappe.new_doc('Notification Log')
            notification.for_user = self.user
            notification.set('type', 'Alert')
            notification.document_type = self.reminder_doctype
            notification.document_name = self.reminder_docname
            notification.subject = self.description
            notification.insert()
        except Exception:
            self.log_error('Failed to send reminder')

@frappe.whitelist()
def create_new_reminder(remind_at: str, description: str, reminder_doctype: str | None=None, reminder_docname: str | None=None):
    if False:
        print('Hello World!')
    reminder = frappe.new_doc('Reminder')
    reminder.description = description
    reminder.remind_at = remind_at
    reminder.reminder_doctype = reminder_doctype
    reminder.reminder_docname = reminder_docname
    return reminder.insert()

def send_reminders():
    if False:
        while True:
            i = 10
    job_freq = cint(frappe.get_conf().scheduler_interval) or 240
    upper_threshold = add_to_date(now_datetime(), seconds=job_freq, as_string=True, as_datetime=True)
    lower_threshold = add_to_date(now_datetime(), hours=-8, as_string=True, as_datetime=True)
    pending_reminders = frappe.get_all('Reminder', filters=[('remind_at', '<=', upper_threshold), ('remind_at', '>=', lower_threshold), ('notified', '=', 0)], pluck='name')
    for reminder in pending_reminders:
        frappe.get_doc('Reminder', reminder).send_reminder()
from urllib.parse import quote
from rq import get_current_job
import frappe
from frappe import _
from frappe.desk.doctype.notification_log.notification_log import enqueue_create_notification
from frappe.model.document import Document
from frappe.monitor import add_data_to_monitor
from frappe.utils import now, time_diff_in_seconds
from frappe.utils.data import cint

class SubmissionQueue(Document):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from frappe.types import DF
        created_at: DF.Datetime | None
        ended_at: DF.Datetime | None
        enqueued_by: DF.Data | None
        exception: DF.LongText | None
        job_id: DF.Link | None
        ref_docname: DF.DynamicLink | None
        ref_doctype: DF.Link | None
        status: DF.Literal['Queued', 'Finished', 'Failed']

    @property
    def created_at(self):
        if False:
            while True:
                i = 10
        return self.creation

    @property
    def enqueued_by(self):
        if False:
            return 10
        return self.owner

    @property
    def queued_doc(self):
        if False:
            i = 10
            return i + 15
        return getattr(self, 'to_be_queued_doc', frappe.get_doc(self.ref_doctype, self.ref_docname))

    @staticmethod
    def clear_old_logs(days=30):
        if False:
            print('Hello World!')
        from frappe.query_builder import Interval
        from frappe.query_builder.functions import Now
        table = frappe.qb.DocType('Submission Queue')
        frappe.db.delete(table, filters=table.modified < Now() - Interval(days=days))

    def insert(self, to_be_queued_doc: Document, action: str):
        if False:
            print('Hello World!')
        self.status = 'Queued'
        self.to_be_queued_doc = to_be_queued_doc
        self.action_for_queuing = action
        super().insert(ignore_permissions=True)

    def lock(self):
        if False:
            return 10
        self.queued_doc.lock()

    def unlock(self):
        if False:
            print('Hello World!')
        self.queued_doc.unlock()

    def update_job_id(self, job_id):
        if False:
            while True:
                i = 10
        frappe.db.set_value(self.doctype, self.name, {'job_id': job_id}, update_modified=False)
        frappe.db.commit()

    def after_insert(self):
        if False:
            for i in range(10):
                print('nop')
        self.queue_action('background_submission', to_be_queued_doc=self.queued_doc, action_for_queuing=self.action_for_queuing, timeout=600, enqueue_after_commit=True)

    def background_submission(self, to_be_queued_doc: Document, action_for_queuing: str):
        if False:
            while True:
                i = 10
        self.update_job_id(get_current_job().id)
        _action = action_for_queuing.lower()
        if _action == 'update':
            _action = 'submit'
        try:
            getattr(to_be_queued_doc, _action)()
            add_data_to_monitor(doctype=to_be_queued_doc.doctype, docname=to_be_queued_doc.name, action=_action, execution_time=time_diff_in_seconds(now(), self.created_at), enqueued_by=self.enqueued_by)
            values = {'status': 'Finished'}
        except Exception:
            values = {'status': 'Failed', 'exception': frappe.get_traceback(with_context=True)}
            frappe.db.rollback()
        values['ended_at'] = now()
        frappe.db.set_value(self.doctype, self.name, values, update_modified=False)
        self.notify(values['status'], action_for_queuing)

    def notify(self, submission_status: str, action: str):
        if False:
            print('Hello World!')
        if submission_status == 'Failed':
            doctype = self.doctype
            docname = self.name
            message = _('Action {0} failed on {1} {2}. View it {3}')
        else:
            doctype = self.ref_doctype
            docname = self.ref_docname
            message = _('Action {0} completed successfully on {1} {2}. View it {3}')
        message_replacements = (frappe.bold(action), frappe.bold(str(self.ref_doctype)), frappe.bold(str(self.ref_docname)))
        time_diff = time_diff_in_seconds(now(), self.created_at)
        if cint(time_diff) <= 60:
            frappe.publish_realtime('msgprint', {'message': message.format(*message_replacements, f"<a href='/app/{quote(doctype.lower().replace(' ', '-'))}/{quote(docname)}'><b>here</b></a>"), 'alert': True, 'indicator': 'red' if submission_status == 'Failed' else 'green'}, user=self.enqueued_by)
        else:
            notification_doc = {'type': 'Alert', 'document_type': doctype, 'document_name': docname, 'subject': message.format(*message_replacements, 'here')}
            notify_to = frappe.db.get_value('User', self.enqueued_by, fieldname='email')
            enqueue_create_notification([notify_to], notification_doc)

    @frappe.whitelist()
    def unlock_doc(self):
        if False:
            return 10
        if self.status != 'Queued':
            return
        self.queued_doc.unlock()
        frappe.msgprint(_('Document Unlocked'))

def queue_submission(doc: Document, action: str, alert: bool=True):
    if False:
        print('Hello World!')
    queue = frappe.new_doc('Submission Queue')
    queue.ref_doctype = doc.doctype
    queue.ref_docname = doc.name
    queue.insert(doc, action)
    if alert:
        frappe.msgprint(_('Queued for Submission. You can track the progress over {0}.').format(f"<a href='/app/submission-queue/{queue.name}'><b>here</b></a>"), indicator='green', alert=True)

@frappe.whitelist()
def get_latest_submissions(doctype, docname):
    if False:
        for i in range(10):
            print('nop')
    latest_submission = frappe.db.get_value('Submission Queue', filters={'ref_doctype': doctype, 'ref_docname': docname}, fieldname=['name', 'exception', 'status'])
    out = None
    if latest_submission:
        out = {'latest_submission': latest_submission[0], 'exc': format_tb(latest_submission[1]), 'status': latest_submission[2]}
    return out

def format_tb(traceback: str | None=None):
    if False:
        i = 10
        return i + 15
    if not traceback:
        return
    return traceback.strip().split('\n')[-1]
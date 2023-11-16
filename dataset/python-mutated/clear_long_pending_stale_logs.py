import frappe
from frappe.core.doctype.log_settings.log_settings import clear_log_table
from frappe.utils import add_to_date, today

def execute():
    if False:
        print('Hello World!')
    'Due to large size of log tables on old sites some table cleanups never finished during daily log clean up. This patch discards such data by using "big delete" code.\n\n\tref: https://github.com/frappe/frappe/issues/16971\n\t'
    DOCTYPE_RETENTION_MAP = {'Error Log': get_current_setting('clear_error_log_after') or 90, 'Activity Log': get_current_setting('clear_activity_log_after') or 90, 'Email Queue': get_current_setting('clear_email_queue_after') or 30, 'Email Queue Recipient': get_current_setting('clear_email_queue_after') or 30, 'Scheduled Job Log': 90}
    for (doctype, retention) in DOCTYPE_RETENTION_MAP.items():
        if is_log_cleanup_stuck(doctype, retention):
            print(f'Clearing old {doctype} records')
            clear_log_table(doctype, retention)

def is_log_cleanup_stuck(doctype: str, retention: int) -> bool:
    if False:
        print('Hello World!')
    'Check if doctype has data significantly older than configured cleanup period'
    threshold = add_to_date(today(), days=retention * -2)
    return bool(frappe.db.exists(doctype, {'modified': ('<', threshold)}))

def get_current_setting(fieldname):
    if False:
        i = 10
        return i + 15
    try:
        return frappe.db.get_single_value('Log Settings', fieldname)
    except Exception:
        pass
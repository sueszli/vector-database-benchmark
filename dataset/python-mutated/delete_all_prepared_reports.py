import frappe

def execute():
    if False:
        for i in range(10):
            print('nop')
    if frappe.db.table_exists('Prepared Report'):
        frappe.reload_doc('core', 'doctype', 'prepared_report')
        prepared_reports = frappe.get_all('Prepared Report')
        for report in prepared_reports:
            frappe.delete_doc('Prepared Report', report.name)
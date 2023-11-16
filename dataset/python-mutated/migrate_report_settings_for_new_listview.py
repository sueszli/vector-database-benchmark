import json
import frappe

def execute():
    if False:
        while True:
            i = 10
    '\n\tMigrate JSON field of Report according to changes in New ListView\n\tRename key columns to fields\n\tRename key add_total_row to add_totals_row\n\tConvert sort_by and sort_order to order_by\n\t'
    reports = frappe.get_all('Report', {'report_type': 'Report Builder'})
    for report_name in reports:
        settings = frappe.db.get_value('Report', report_name, 'json')
        if not settings:
            continue
        settings = frappe._dict(json.loads(settings))
        settings.fields = settings.columns or []
        settings.pop('columns', None)
        settings.order_by = (settings.sort_by or 'modified') + ' ' + (settings.order_by or 'desc')
        settings.add_totals_row = settings.add_total_row
        settings.pop('add_total_row', None)
        frappe.db.set_value('Report', report_name, 'json', json.dumps(settings))
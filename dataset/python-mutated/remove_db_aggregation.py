import re
import frappe
from frappe.query_builder import DocType

def execute():
    if False:
        print('Hello World!')
    'Replace temporarily available Database Aggregate APIs on frappe (develop)\n\n\tAPIs changed:\n\t        * frappe.db.max => frappe.qb.max\n\t        * frappe.db.min => frappe.qb.min\n\t        * frappe.db.sum => frappe.qb.sum\n\t        * frappe.db.avg => frappe.qb.avg\n\t'
    ServerScript = DocType('Server Script')
    server_scripts = frappe.qb.from_(ServerScript).where(ServerScript.script.like('%frappe.db.max(%') | ServerScript.script.like('%frappe.db.min(%') | ServerScript.script.like('%frappe.db.sum(%') | ServerScript.script.like('%frappe.db.avg(%')).select('name', 'script').run(as_dict=True)
    for server_script in server_scripts:
        (name, script) = (server_script['name'], server_script['script'])
        for agg in ['avg', 'max', 'min', 'sum']:
            script = re.sub(f'frappe.db.{agg}\\(', f'frappe.qb.{agg}(', script)
        frappe.db.set_value('Server Script', name, 'script', script)
import json
import frappe

def execute():
    if False:
        return 10
    'Convert Query Report json to support other content'
    records = frappe.get_all('Report', filters={'json': ['!=', '']}, fields=['name', 'json'])
    for record in records:
        jstr = record['json']
        data = json.loads(jstr)
        if isinstance(data, list):
            jstr = f'{{"columns":{jstr}}}'
            frappe.db.set_value('Report', record['name'], 'json', jstr)
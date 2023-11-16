import frappe

def execute():
    if False:
        print('Hello World!')
    duplicateRecords = frappe.db.sql('select count(name) as `count`, allow, user, for_value\n\t\tfrom `tabUser Permission`\n\t\tgroup by allow, user, for_value\n\t\thaving count(*) > 1 ', as_dict=1)
    for record in duplicateRecords:
        frappe.db.sql('delete from `tabUser Permission`\n\t\t\twhere allow=%s and user=%s and for_value=%s limit {}'.format(record.count - 1), (record.allow, record.user, record.for_value))
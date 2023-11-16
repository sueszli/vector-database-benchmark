import frappe

def execute():
    if False:
        print('Hello World!')
    frappe.db.delete('Series', {'current': 0})
    duplicate_keys = frappe.db.sql('\n\t\tSELECT name, max(current) as current\n\t\tfrom\n\t\t\t`tabSeries`\n\t\tgroup by\n\t\t\tname\n\t\thaving count(name) > 1\n\t', as_dict=True)
    for row in duplicate_keys:
        frappe.db.delete('Series', {'name': row.name})
        if row.current:
            frappe.db.sql('insert into `tabSeries`(`name`, `current`) values (%(name)s, %(current)s)', row)
    frappe.db.commit()
    frappe.db.sql('ALTER table `tabSeries` ADD PRIMARY KEY IF NOT EXISTS (name)')
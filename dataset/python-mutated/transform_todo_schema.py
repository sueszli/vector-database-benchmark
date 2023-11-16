import frappe
from frappe.query_builder.utils import DocType

def execute():
    if False:
        i = 10
        return i + 15
    ToDo = DocType('ToDo')
    frappe.reload_doctype('ToDo', force=True)
    frappe.qb.update(ToDo).set(ToDo.allocated_to, ToDo.owner).run()
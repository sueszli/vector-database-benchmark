import frappe
from frappe.query_builder import DocType

def execute():
    if False:
        while True:
            i = 10
    RATING_FIELD_TYPE = 'decimal(3,2)'
    rating_fields = frappe.get_all('DocField', fields=['parent', 'fieldname'], filters={'fieldtype': 'Rating'})
    custom_rating_fields = frappe.get_all('Custom Field', fields=['dt', 'fieldname'], filters={'fieldtype': 'Rating'})
    for _field in rating_fields + custom_rating_fields:
        doctype_name = _field.get('parent') or _field.get('dt')
        doctype = DocType(doctype_name)
        field = _field.fieldname
        if frappe.conf.db_type == 'mariadb' and frappe.db.get_column_type(doctype_name, field) == RATING_FIELD_TYPE:
            continue
        frappe.db.commit()
        frappe.db.change_column_type(doctype_name, column=field, type=RATING_FIELD_TYPE, nullable=True)
        frappe.qb.update(doctype).set(doctype[field], doctype[field] / 5).run()
        frappe.db.commit()
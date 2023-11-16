import frappe

def execute():
    if False:
        return 10
    frappe.reload_doc('core', 'doctype', 'docfield', force=True)
    frappe.reload_doc('custom', 'doctype', 'custom_field', force=True)
    frappe.reload_doc('custom', 'doctype', 'customize_form_field', force=True)
    frappe.reload_doc('custom', 'doctype', 'property_setter', force=True)
    frappe.db.sql("\n\t\tupdate `tabDocField`\n\t\tset fetch_from = options, options=''\n\t\twhere options like '%.%' and (fetch_from is NULL OR fetch_from='')\n \t\tand fieldtype in ('Data', 'Read Only', 'Text', 'Small Text', 'Text Editor', 'Code', 'Link', 'Check')\n \t\tand fieldname!='naming_series'\n\t")
    frappe.db.sql("\n\t\tupdate `tabCustom Field`\n\t\tset fetch_from = options, options=''\n\t\twhere options like '%.%' and (fetch_from is NULL OR fetch_from='')\n \t\tand fieldtype in ('Data', 'Read Only', 'Text', 'Small Text', 'Text Editor', 'Code', 'Link', 'Check')\n \t\tand fieldname!='naming_series'\n\t")
    frappe.db.sql('\n\t\tupdate `tabProperty Setter`\n\t\tset property="fetch_from", name=concat(doc_type, \'-\', field_name, \'-\', property)\n\t\twhere property="options" and value like \'%.%\'\n\t\tand property_type in (\'Data\', \'Read Only\', \'Text\', \'Small Text\', \'Text Editor\', \'Code\', \'Link\', \'Check\')\n\t\tand field_name!=\'naming_series\'\n\t')
import frappe
dynamic_link_queries = ["select `tabDocField`.parent,\n\t\t`tabDocType`.read_only, `tabDocType`.in_create,\n\t\t`tabDocField`.fieldname, `tabDocField`.options\n\tfrom `tabDocField`, `tabDocType`\n\twhere `tabDocField`.fieldtype='Dynamic Link' and\n\t`tabDocType`.`name`=`tabDocField`.parent\n\torder by `tabDocType`.read_only, `tabDocType`.in_create", "select `tabCustom Field`.dt as parent,\n\t\t`tabDocType`.read_only, `tabDocType`.in_create,\n\t\t`tabCustom Field`.fieldname, `tabCustom Field`.options\n\tfrom `tabCustom Field`, `tabDocType`\n\twhere `tabCustom Field`.fieldtype='Dynamic Link' and\n\t`tabDocType`.`name`=`tabCustom Field`.dt\n\torder by `tabDocType`.read_only, `tabDocType`.in_create"]

def get_dynamic_link_map(for_delete=False):
    if False:
        i = 10
        return i + 15
    'Build a map of all dynamically linked tables. For example,\n\t        if Note is dynamically linked to ToDo, the function will return\n\t        `{"Note": ["ToDo"], "Sales Invoice": ["Journal Entry Detail"]}`\n\n\tNote: Will not map single doctypes\n\t'
    if getattr(frappe.local, 'dynamic_link_map', None) is None or frappe.flags.in_test:
        dynamic_link_map = {}
        for df in get_dynamic_links():
            meta = frappe.get_meta(df.parent)
            if meta.issingle:
                dynamic_link_map.setdefault(meta.name, []).append(df)
            else:
                try:
                    links = frappe.db.sql_list('select distinct {options} from `tab{parent}`'.format(**df))
                    for doctype in links:
                        dynamic_link_map.setdefault(doctype, []).append(df)
                except frappe.db.TableMissingError:
                    pass
        frappe.local.dynamic_link_map = dynamic_link_map
    return frappe.local.dynamic_link_map

def get_dynamic_links():
    if False:
        print('Hello World!')
    'Return list of dynamic link fields as DocField.\n\tUses cache if possible'
    df = []
    for query in dynamic_link_queries:
        df += frappe.db.sql(query, as_dict=True)
    return df
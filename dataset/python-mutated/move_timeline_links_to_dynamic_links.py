import frappe

def execute():
    if False:
        for i in range(10):
            print('nop')
    communications = frappe.db.sql("\n\t\tSELECT\n\t\t\t`tabCommunication`.name, `tabCommunication`.creation, `tabCommunication`.modified,\n\t\t\t`tabCommunication`.modified_by,`tabCommunication`.timeline_doctype, `tabCommunication`.timeline_name,\n\t\t\t`tabCommunication`.link_doctype, `tabCommunication`.link_name\n\t\tFROM `tabCommunication`\n\t\tWHERE `tabCommunication`.communication_medium='Email'\n\t", as_dict=True)
    name = 1000000000
    values = []
    for (count, communication) in enumerate(communications):
        counter = 1
        if communication.timeline_doctype and communication.timeline_name:
            name += 1
            values.append('({}, "{}", "timeline_links", "Communication", "{}", "{}", "{}", "{}", "{}", "{}")'.format(counter, str(name), frappe.db.escape(communication.name), frappe.db.escape(communication.timeline_doctype), frappe.db.escape(communication.timeline_name), communication.creation, communication.modified, communication.modified_by))
            counter += 1
        if communication.link_doctype and communication.link_name:
            name += 1
            values.append('({}, "{}", "timeline_links", "Communication", "{}", "{}", "{}", "{}", "{}", "{}")'.format(counter, str(name), frappe.db.escape(communication.name), frappe.db.escape(communication.link_doctype), frappe.db.escape(communication.link_name), communication.creation, communication.modified, communication.modified_by))
        if values and (count % 10000 == 0 or count == len(communications) - 1):
            frappe.db.sql('\n\t\t\t\tINSERT INTO `tabCommunication Link`\n\t\t\t\t\t(`idx`, `name`, `parentfield`, `parenttype`, `parent`, `link_doctype`, `link_name`, `creation`,\n\t\t\t\t\t`modified`, `modified_by`)\n\t\t\t\tVALUES {}\n\t\t\t'.format(', '.join([d for d in values])))
            values = []
    frappe.db.add_index('Communication Link', ['link_doctype', 'link_name'])
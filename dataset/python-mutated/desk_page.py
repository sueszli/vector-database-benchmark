import frappe

@frappe.whitelist()
def get(name):
    if False:
        while True:
            i = 10
    '\n\tReturn the :term:`doclist` of the `Page` specified by `name`\n\t'
    page = frappe.get_doc('Page', name)
    if page.is_permitted():
        page.load_assets()
        docs = frappe._dict(page.as_dict())
        if getattr(page, '_dynamic_page', None):
            docs['_dynamic_page'] = 1
        return docs
    else:
        frappe.response['403'] = 1
        raise frappe.PermissionError('No read permission for Page %s' % (page.title or name))

@frappe.whitelist(allow_guest=True)
def getpage():
    if False:
        for i in range(10):
            print('nop')
    '\n\tLoad the page from `frappe.form` and send it via `frappe.response`\n\t'
    page = frappe.form_dict.get('name')
    doc = get(page)
    frappe.response.docs.append(doc)
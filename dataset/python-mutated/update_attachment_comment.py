import frappe

def execute():
    if False:
        return 10
    frappe.db.auto_commit_on_many_writes = 1
    for (name, content) in frappe.get_all('Comment', filters={'comment_type': 'Attachment'}, fields=['name', 'content'], as_list=True):
        if not content:
            continue
        start = content.find('<a href')
        if start != -1:
            content = content[start:]
        end = content.find('</i>')
        end = content.find('</a>') if end == -1 else end
        if end != -1:
            content = content[:end + 4]
        frappe.db.set_value('Comment', name, 'content', content, update_modified=False)
    for (name, content) in frappe.get_all('Comment', filters={'comment_type': 'Attachment Removed'}, fields=['name', 'content'], as_list=True):
        if content and content.startswith('Removed '):
            frappe.db.set_value('Comment', name, 'content', content[8:], update_modified=False)
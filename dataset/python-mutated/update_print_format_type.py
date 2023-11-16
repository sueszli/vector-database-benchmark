import frappe

def execute():
    if False:
        print('Hello World!')
    frappe.db.sql("\n\t\tUPDATE `tabPrint Format`\n\t\tSET `print_format_type` = 'Jinja'\n\t\tWHERE `print_format_type` in ('Server', 'Client')\n\t")
    frappe.db.sql("\n\t\tUPDATE `tabPrint Format`\n\t\tSET `print_format_type` = 'JS'\n\t\tWHERE `print_format_type` = 'Js'\n\t")
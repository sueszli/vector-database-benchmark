import frappe

def execute():
    if False:
        for i in range(10):
            print('nop')
    frappe.reload_doc('website', 'doctype', 'website_theme_ignore_app')
    frappe.reload_doc('website', 'doctype', 'color')
    frappe.reload_doc('website', 'doctype', 'website_theme', force=True)
    for theme in frappe.get_all('Website Theme'):
        doc = frappe.get_doc('Website Theme', theme.name)
        setup_color_record(doc)
        if not doc.get('custom_scss') and doc.theme_scss:
            doc.custom_scss = doc.theme_scss
            doc.save()

def setup_color_record(doc):
    if False:
        while True:
            i = 10
    color_fields = ['primary_color', 'text_color', 'light_color', 'dark_color', 'background_color']
    for color_field in color_fields:
        color_code = doc.get(color_field)
        if not color_code or frappe.db.exists('Color', color_code):
            continue
        frappe.get_doc({'doctype': 'Color', '__newname': color_code, 'color': color_code}).insert()
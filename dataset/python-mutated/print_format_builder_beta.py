import functools
import frappe

@frappe.whitelist()
def get_google_fonts():
    if False:
        print('Hello World!')
    return _get_google_fonts()

@functools.lru_cache
def _get_google_fonts():
    if False:
        return 10
    file_path = frappe.get_app_path('frappe', 'data', 'google_fonts.json')
    return frappe.parse_json(frappe.read_file(file_path))
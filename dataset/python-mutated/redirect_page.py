import frappe
from frappe.website.utils import build_response

class RedirectPage:

    def __init__(self, path, http_status_code=301):
        if False:
            return 10
        self.path = path
        self.http_status_code = http_status_code

    def can_render(self):
        if False:
            print('Hello World!')
        return True

    def render(self):
        if False:
            print('Hello World!')
        return build_response(self.path, '', 301, {'Location': frappe.flags.redirect_location or (frappe.local.response or {}).get('location'), 'Cache-Control': 'no-store, no-cache, must-revalidate'})
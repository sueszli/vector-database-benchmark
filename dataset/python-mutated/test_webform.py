import frappe
from frappe.tests.utils import FrappeTestCase
from frappe.utils import set_request
from frappe.website.serve import get_response
from frappe.www.list import get_list_context

class TestWebform(FrappeTestCase):

    def test_webform_publish_functionality(self):
        if False:
            for i in range(10):
                print('nop')
        request_data = frappe.get_doc('Web Form', 'request-data')
        request_data.published = True
        request_data.save()
        set_request(method='GET', path='request-data/new')
        response = get_response()
        self.assertEqual(response.status_code, 200)
        request_data.published = False
        request_data.save()
        response = get_response()
        self.assertEqual(response.status_code, 404)

    def test_get_context_hook_of_webform(self):
        if False:
            return 10
        create_custom_doctype()
        create_webform()
        context_list = get_list_context('', 'Custom Doctype', 'test-webform')
        self.assertFalse(context_list)
        set_webform_hook('webform_list_context', 'frappe.www._test._test_webform.webform_list_context')
        context_list = get_list_context('', 'Custom Doctype', 'test-webform')
        self.assertTrue(context_list)

def create_custom_doctype():
    if False:
        for i in range(10):
            print('nop')
    frappe.get_doc({'doctype': 'DocType', 'name': 'Custom Doctype', 'module': 'Core', 'custom': 1, 'fields': [{'label': 'Title', 'fieldname': 'title', 'fieldtype': 'Data'}]}).insert(ignore_if_duplicate=True)

def create_webform():
    if False:
        for i in range(10):
            print('nop')
    frappe.get_doc({'doctype': 'Web Form', 'module': 'Core', 'title': 'Test Webform', 'route': 'test-webform', 'doc_type': 'Custom Doctype', 'web_form_fields': [{'doctype': 'Web Form Field', 'fieldname': 'title', 'fieldtype': 'Data', 'label': 'Title'}]}).insert(ignore_if_duplicate=True)

def set_webform_hook(key, value):
    if False:
        return 10
    from frappe import hooks
    for hook in 'webform_list_context':
        if hasattr(hooks, hook):
            delattr(hooks, hook)
    setattr(hooks, key, value)
    frappe.cache.delete_key('app_hooks')